// Gemma 4 in the browser (WebGPU) through the MediaPipe LLM Inference task — adapted from the
// LiteRT samples SAM 2 web demo (google-ai-edge/litert-samples, models/sam2/.../web_demo/app/src/
// gemma_web.ts, Apache-2.0).
//
// Everything heavy runs in gemma.worker.ts (model download/cache streaming, WASM init, token decoding),
// so loading a 2–3 GB model never freezes the camera/relight loop on the main thread.
//
// The models are the GPU ("hw") .litertlm builds: text decoder + vision encoder/adapter sections.
// public/genai/ carries the MediaPipe GenAI build from that sample which also reads the vision sections.
import type { GemmaWorkerRequest, GemmaWorkerResponse } from './gemma.worker.ts';

export interface GemmaModel {
  id: string;
  name: string;
  url: string;
  bytes: number;
}

const BUCKET = 'https://storage.googleapis.com/litertjs_demos/';
export const GEMMA_MODELS: GemmaModel[] = [
  { id: 'gemma4-e2b', name: 'Gemma 4 E2B', bytes: 2237340276, url: `${BUCKET}gemma_4_models/gemma4-e2b-hw.litertlm` },
  { id: 'gemma4-e4b', name: 'Gemma 4 E4B', bytes: 3201116788, url: `${BUCKET}gemma_4_models/gemma4-e4b-hw-int4-20260622.litertlm` },
];
const GENAI_WASM = `${BUCKET}wasm/genai_wasm_internal.wasm`;
const CACHE_NAME = 'meet-relight-gemma4';

export const gemmaSupported = () => 'gpu' in navigator && typeof caches !== 'undefined';

/** True if `id` is already in the browser cache (load will take seconds, not minutes). */
export async function gemmaCached(id: string): Promise<boolean> {
  const m = GEMMA_MODELS.find((x) => x.id === id);
  if (!m || typeof caches === 'undefined') return false;
  const c = await caches.open(CACHE_NAME).catch(() => null);
  return !!(await c?.match(m.url).catch(() => undefined));
}

export type Progress = (text: string, fraction?: number) => void;

/** Handle to the Gemma worker. */
export class GemmaClient {
  private worker: Worker | null = null;
  private loadedId: string | null = null;
  private loading: Promise<void> | null = null;
  private nextId = 1;
  private pending = new Map<number, { resolve: (t: string) => void; reject: (e: Error) => void; onPartial?: (t: string) => void }>();
  private onProgress: Progress = () => {};

  constructor(private assetBase: string) {}

  get ready(): boolean {
    return !!this.loadedId && !this.loading;
  }

  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    const w = new Worker(new URL('./gemma.worker.ts', import.meta.url), { type: 'module', name: 'gemma' });
    w.onmessage = (e: MessageEvent<GemmaWorkerResponse>) => {
      const m = e.data;
      if (m.type === 'progress') this.onProgress(m.text, m.fraction);
      else if (m.type === 'partial') this.pending.get(m.id)?.onPartial?.(m.text);
      else if (m.type === 'done') {
        this.pending.get(m.id)?.resolve(m.text);
        this.pending.delete(m.id);
      } else if (m.type === 'error' && m.id !== undefined) {
        this.pending.get(m.id)?.reject(new Error(m.message));
        this.pending.delete(m.id);
      }
    };
    this.worker = w;
    return w;
  }

  /** Load (or switch to) a model. Resolves when it is ready to generate. */
  load(id: string, progress: Progress): Promise<void> {
    if (this.loadedId === id && this.loading) return this.loading;
    if (this.loadedId === id) return Promise.resolve();
    const m = GEMMA_MODELS.find((x) => x.id === id);
    if (!m) return Promise.reject(new Error(`unknown model ${id}`));
    this.onProgress = progress;
    const w = this.ensureWorker();
    this.loadedId = id;
    this.loading = new Promise<void>((resolve, reject) => {
      const onMsg = (e: MessageEvent<GemmaWorkerResponse>) => {
        if (e.data.type === 'ready') {
          w.removeEventListener('message', onMsg);
          progress(`${m.name} ready (${e.data.loadSeconds.toFixed(0)} s to load).`, 1);
          resolve();
        } else if (e.data.type === 'error' && e.data.id === undefined) {
          w.removeEventListener('message', onMsg);
          reject(new Error(e.data.message));
        }
      };
      w.addEventListener('message', onMsg);
      w.postMessage({
        type: 'load', url: m.url, bytes: m.bytes, name: m.name,
        genaiBase: new URL('genai/', this.assetBase).href, wasmUrl: GENAI_WASM, cacheName: CACHE_NAME,
      } satisfies GemmaWorkerRequest);
    });
    this.loading.then(() => { this.loading = null; }, () => { this.loading = null; this.loadedId = null; });
    return this.loading;
  }

  /** One user turn (optionally with an image), streamed. */
  async turn(prompt: string, image: ImageBitmap | null, onPartial?: (text: string) => void): Promise<string> {
    const w = this.ensureWorker();
    const id = this.nextId++;
    return new Promise<string>((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onPartial });
      w.postMessage({ type: 'generate', id, prompt, image } satisfies GemmaWorkerRequest, image ? [image] : []);
    });
  }

  cancel(): void {
    this.worker?.postMessage({ type: 'cancel' } satisfies GemmaWorkerRequest);
  }
}

/** Downscale an image (longest side ≤ maxSide) to a transferable ImageBitmap for a prompt. */
export function toPromptImage(src: ImageBitmap, maxSide = 768): Promise<ImageBitmap> {
  const s = Math.min(1, maxSide / Math.max(src.width, src.height, 1));
  return createImageBitmap(src, { resizeWidth: Math.max(1, Math.round(src.width * s)), resizeHeight: Math.max(1, Math.round(src.height * s)), resizeQuality: 'high' });
}
