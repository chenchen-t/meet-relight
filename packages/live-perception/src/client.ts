import type { WorkerRequest, WorkerResponse, PerceptionResult } from './protocol.ts';

/**
 * Main-thread handle to the perception worker. Up to `maxInFlight` frames are
 * pipelined (default 2): while the worker infers frame N, frame N+1 is already
 * downscaled and queued, so throughput ≈ 1 / inference-time instead of
 * 1 / (inference + transfer + scheduling). This keeps up with 60 fps cameras so
 * frame-synced display does not drop to 30 fps.
 */
export class PerceptionClient {
  maxInFlight = 2;
  inFlight = 0;
  onResult: (r: PerceptionResult) => void = () => {};
  onError: (msg: string) => void = (m) => console.error('[perception]', m);
  /** Round-trip latency (submit → result) of the last frame, ms. */
  latencyMs = 0;
  private sentAt = new Map<number, number>();
  private lastTs = 0;

  private constructor(private worker: Worker) {
    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const m = e.data;
      if (m.type === 'result') {
        this.inFlight = Math.max(0, this.inFlight - 1);
        const t = this.sentAt.get(m.id);
        if (t !== undefined) this.latencyMs = performance.now() - t;
        this.sentAt.delete(m.id);
        this.onResult(m);
      } else if (m.type === 'error') {
        this.inFlight = Math.max(0, this.inFlight - 1);
        this.onError(m.message);
      }
    };
  }

  get busy(): boolean {
    return this.inFlight >= this.maxInFlight;
  }

  static async create(base = new URL(import.meta.env?.BASE_URL ?? '/', location.href).href): Promise<PerceptionClient> {
    const worker = new Worker(new URL('./perception.worker.ts', import.meta.url), { type: 'module' });
    await new Promise<void>((resolve, reject) => {
      const onMsg = (e: MessageEvent<WorkerResponse>) => {
        if (e.data.type === 'ready') {
          worker.removeEventListener('message', onMsg);
          resolve();
        } else if (e.data.type === 'error') reject(new Error(e.data.message));
      };
      worker.addEventListener('message', onMsg);
      worker.onerror = (e) => reject(new Error(`perception worker failed: ${e.message}`));
      worker.postMessage({ type: 'init', base } satisfies WorkerRequest);
    });
    return new PerceptionClient(worker);
  }

  /** Downscale `source` to procW×procH and send to the worker. Returns false if the pipeline is full. */
  /** When false the worker runs face landmarks only (segmentation done elsewhere, e.g. LiteRT.js on WebGPU). */
  segment = true;

  async submit(source: CanvasImageSource & ImageBitmapSource, id: number, procW: number, procH: number): Promise<boolean> {
    if (this.busy) return false;
    this.inFlight++;
    try {
      const bitmap = await createImageBitmap(source, { resizeWidth: procW, resizeHeight: procH, resizeQuality: 'medium' });
      // MediaPipe VIDEO mode requires strictly increasing timestamps.
      const ts = Math.max(performance.now(), this.lastTs + 0.001);
      this.lastTs = ts;
      this.sentAt.set(id, performance.now());
      this.worker.postMessage({ type: 'frame', id, bitmap, timestampMs: ts, segment: this.segment } satisfies WorkerRequest, [bitmap]);
      return true;
    } catch (e) {
      this.inFlight--;
      throw e;
    }
  }
}
