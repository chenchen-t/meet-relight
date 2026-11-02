import gatherSrc from './gather.wgsl?raw';
import { SAMPLE_STRIDE } from './shFit.ts';

export type GatherSourceKind = 'external' | 'texture';

/**
 * GPU → CPU sample gatherer. `encode()` records the gather + copy; after
 * submit, `read()` resolves with packed samples [n.xyz, rgb] of valid cells.
 * Readback is asynchronous (mapAsync) — never stalls the render loop.
 */
export class SampleGatherer {
  private pipe: GPUComputePipeline;
  private sampler: GPUSampler;
  private ubuf: GPUBuffer;
  private sbuf!: GPUBuffer;
  private readBufs: GPUBuffer[] = [];
  private cells = 0;
  private gridW = 0;
  private gridH = 0;
  pending = 0;

  constructor(
    private device: GPUDevice,
    kind: GatherSourceKind,
    private stride = 3,
  ) {
    const code = gatherSrc
      .replace(
        '{{SRC_DECL}}',
        kind === 'external'
          ? '@group(0) @binding(1) var srcTex: texture_external;'
          : '@group(0) @binding(1) var srcTex: texture_2d<f32>;',
      )
      .replace(
        '{{SRC_LOAD}}',
        kind === 'external'
          ? 'textureSampleBaseClampToEdge(srcTex, samp, uv).rgb'
          : 'textureSampleLevel(srcTex, samp, uv, 0.0).rgb',
      );
    if (code.includes('{{')) throw new Error('gather.wgsl: unresolved template token');
    this.pipe = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ label: 'gather', code }), entryPoint: 'main' },
    });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.ubuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  private ensure(geoW: number, geoH: number): void {
    const gw = Math.floor(geoW / this.stride);
    const gh = Math.floor(geoH / this.stride);
    if (gw === this.gridW && gh === this.gridH) return;
    this.gridW = gw;
    this.gridH = gh;
    this.cells = gw * gh;
    this.sbuf?.destroy();
    for (const b of this.readBufs) b.destroy();
    this.readBufs = [];
    this.sbuf = this.device.createBuffer({ size: this.cells * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.device.queue.writeBuffer(this.ubuf, 0, new Uint32Array([gw, gh, this.stride, 0]));
  }

  /** Records the gather. Returns a function to call after queue.submit() that resolves with samples. */
  encode(
    encoder: GPUCommandEncoder,
    source: GPUExternalTexture | GPUTextureView,
    geo: GPUTexture,
    mask: GPUTextureView,
  ): () => Promise<Float32Array> {
    this.ensure(geo.width, geo.height);
    const bg = this.device.createBindGroup({
      layout: this.pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: source },
        { binding: 2, resource: geo.createView() },
        { binding: 3, resource: mask },
        { binding: 4, resource: { buffer: this.sbuf } },
        { binding: 5, resource: { buffer: this.ubuf } },
      ],
    });
    const cp = encoder.beginComputePass();
    cp.setPipeline(this.pipe);
    cp.setBindGroup(0, bg);
    cp.dispatchWorkgroups(Math.ceil(this.gridW / 8), Math.ceil(this.gridH / 8));
    cp.end();
    const read =
      this.readBufs.pop() ??
      this.device.createBuffer({ size: this.cells * 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(this.sbuf, 0, read, 0, this.cells * 32);
    this.pending++;
    const cells = this.cells;
    return async () => {
      try {
        await read.mapAsync(GPUMapMode.READ);
        const f = new Float32Array(read.getMappedRange());
        let valid = 0;
        for (let i = 0; i < cells; i++) if (f[i * 8 + 3]! > 0.5) valid++;
        const out = new Float32Array(valid * SAMPLE_STRIDE);
        let k = 0;
        for (let i = 0; i < cells; i++) {
          const o = i * 8;
          if (f[o + 3]! <= 0.5) continue;
          out[k++] = f[o]!;
          out[k++] = f[o + 1]!;
          out[k++] = f[o + 2]!;
          out[k++] = f[o + 4]!;
          out[k++] = f[o + 5]!;
          out[k++] = f[o + 6]!;
        }
        read.unmap();
        if (read.size === cells * 32) this.readBufs.push(read);
        else read.destroy();
        return out;
      } finally {
        this.pending--;
      }
    };
  }
}
