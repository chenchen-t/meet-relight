/**
 * Minimal WebGPU helpers shared by every GPU package.
 */

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  hasTimestamps: boolean;
  hasFloat32Filterable: boolean;
}

export async function initGpu(): Promise<GpuContext> {
  if (!('gpu' in navigator)) throw new Error('WebGPU is not supported in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter available.');
  const want: GPUFeatureName[] = ['timestamp-query', 'float32-filterable'];
  const requiredFeatures = want.filter((f) => adapter.features.has(f));
  const device = await adapter.requestDevice({ requiredFeatures });
  device.lost.then((info) => console.error('WebGPU device lost:', info.message));
  return {
    adapter,
    device,
    hasTimestamps: device.features.has('timestamp-query'),
    hasFloat32Filterable: device.features.has('float32-filterable'),
  };
}

export function configureCanvas(device: GPUDevice, canvas: HTMLCanvasElement): {
  context: GPUCanvasContext;
  format: GPUTextureFormat;
} {
  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('Failed to get webgpu canvas context.');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });
  return { context, format };
}

/**
 * GPU pass timer based on timestamp queries. Results arrive asynchronously
 * (a few frames late); read `lastMs` / `avgMs`.
 */
export class GpuTimer {
  private querySet?: GPUQuerySet;
  private resolveBuf?: GPUBuffer;
  private readBufs: GPUBuffer[] = [];
  lastMs = 0;
  avgMs = 0;

  constructor(private device: GPUDevice, enabled: boolean) {
    if (!enabled) return;
    this.querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
    this.resolveBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
  }

  get enabled(): boolean {
    return !!this.querySet;
  }

  /** Spread into a GPURenderPassDescriptor / GPUComputePassDescriptor. */
  timestampWrites(): { timestampWrites?: GPURenderPassTimestampWrites } {
    if (!this.querySet) return {};
    return {
      timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
    };
  }

  /** Call after the timed pass, before encoder.finish(). Returns a callback to run after submit. */
  resolve(encoder: GPUCommandEncoder): () => void {
    if (!this.querySet || !this.resolveBuf) return () => {};
    const read =
      this.readBufs.pop() ??
      this.device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.resolveQuerySet(this.querySet, 0, 2, this.resolveBuf, 0);
    encoder.copyBufferToBuffer(this.resolveBuf, 0, read, 0, 16);
    return () => {
      read.mapAsync(GPUMapMode.READ).then(
        () => {
          const t = new BigInt64Array(read.getMappedRange());
          const ms = Number(t[1]! - t[0]!) / 1e6;
          read.unmap();
          this.readBufs.push(read);
          if (ms >= 0 && ms < 1000) {
            this.lastMs = ms;
            this.avgMs = this.avgMs === 0 ? ms : this.avgMs * 0.95 + ms * 0.05;
          }
        },
        () => {},
      );
    };
  }
}

/** Rolling FPS / CPU-time meter. */
export class FrameMeter {
  private last = performance.now();
  fps = 0;
  tick(): void {
    const now = performance.now();
    const dt = now - this.last;
    this.last = now;
    const inst = dt > 0 ? 1000 / dt : 0;
    this.fps = this.fps === 0 ? inst : this.fps * 0.95 + inst * 0.05;
  }
}

export class RollingAvg {
  value = 0;
  constructor(private alpha = 0.05) {}
  push(v: number): number {
    this.value = this.value === 0 ? v : this.value * (1 - this.alpha) + v * this.alpha;
    return this.value;
  }
}
