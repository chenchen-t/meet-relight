/// <reference lib="webworker" />
// Gemma 4 runs in this worker so the multi-GB model stream, WASM init and token decoding never block
// the main thread (camera → relight loop). WebGPU and the Cache API are both available in workers.
//
// MediaPipe's loader calls importScripts(), which throws in module workers (what Vite emits); polyfill
// it with sync XHR + indirect eval (same trick as perception.worker.ts).
declare const self: DedicatedWorkerGlobalScope;

(self as any).importScripts = (...urls: string[]) => {
  for (const url of urls) {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', url, false);
    xhr.send();
    if (xhr.status !== 200) throw new Error(`importScripts polyfill: ${url} → ${xhr.status}`);
    (0, eval)(xhr.responseText + `\n//# sourceURL=${url}`);
  }
};

export type GemmaWorkerRequest =
  | { type: 'load'; url: string; bytes: number; name: string; genaiBase: string; wasmUrl: string; cacheName: string }
  | { type: 'generate'; id: number; prompt: string; image: ImageBitmap | null }
  | { type: 'cancel' };
export type GemmaWorkerResponse =
  | { type: 'progress'; text: string; fraction?: number }
  | { type: 'ready'; loadSeconds: number }
  | { type: 'partial'; id: number; text: string }
  | { type: 'done'; id: number; text: string }
  | { type: 'error'; id?: number; message: string };

interface TrustedUrl { h: unknown }
interface WasmFileset { wasmLoaderPath: TrustedUrl; wasmBinaryPath: TrustedUrl }
type PromptPart = string | { imageSource: CanvasImageSource };
interface LlmInference {
  generateResponse(query: PromptPart[], progress?: (partial: string, done: boolean) => void): Promise<string>;
  cancelProcessing(): void;
  close(): void;
}
interface GenAiModule {
  FilesetResolver: { forGenAiTasks(): Promise<WasmFileset> };
  LlmInference: { createFromOptions(fileset: WasmFileset, options: Record<string, unknown>): Promise<LlmInference> };
}

const post = (m: GemmaWorkerResponse) => self.postMessage(m);
const gb = (n: number) => (n / 1e9).toFixed(2);
let llm: LlmInference | null = null;

function counted(body: ReadableStream<Uint8Array>, total: number, label: (n: number) => string) {
  let n = 0, last = -1;
  return body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, ctl) {
      n += chunk.byteLength;
      const step = Math.floor((n / total) * 200);
      if (step !== last) { last = step; post({ type: 'progress', text: label(n), fraction: n / total }); }
      ctl.enqueue(chunk);
    },
  }));
}

/** Model as a stream: from the Cache API, or downloaded into it first (then streamed back from disk). */
async function modelStream(m: Extract<GemmaWorkerRequest, { type: 'load' }>): Promise<ReadableStream<Uint8Array>> {
  const cache = await caches.open(m.cacheName).catch(() => null);
  const fromDisk = (body: ReadableStream<Uint8Array>, total: number) =>
    counted(body, total, (n) => `Loading ${m.name} onto the GPU: ${gb(n)} / ${gb(total)} GB`);
  const hit = await cache?.match(m.url).catch(() => undefined);
  if (hit?.ok && hit.body) return fromDisk(hit.body, Number(hit.headers.get('Content-Length')) || m.bytes);
  post({ type: 'progress', text: `Downloading ${m.name} (${gb(m.bytes)} GB, once — kept in the browser cache)…`, fraction: 0 });
  const res = await fetch(m.url);
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status} for ${m.url}`);
  const total = Number(res.headers.get('Content-Length')) || m.bytes;
  const download = counted(res.body, total, (n) => `Downloading ${m.name}: ${gb(n)} / ${gb(total)} GB`);
  if (cache) {
    try {
      await cache.put(m.url, new Response(download, { headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(total) } }));
      const back = await cache.match(m.url);
      if (back?.body) return fromDisk(back.body, total);
    } catch (e) {
      console.warn('could not cache model; streaming from the network', e);
    }
  }
  const again = await fetch(m.url);
  if (!again.ok || !again.body) throw new Error(`HTTP ${again.status} for ${m.url}`);
  return counted(again.body, total, (n) => `Downloading ${m.name} (not cached): ${gb(n)} / ${gb(total)} GB`);
}

self.onmessage = async (e: MessageEvent<GemmaWorkerRequest>) => {
  const msg = e.data;
  if (msg.type === 'load') {
    try {
      llm?.close();
      llm = null;
      const genai = await import(/* @vite-ignore */ `${msg.genaiBase}genai_js.js`) as GenAiModule;
      const fileset = await genai.FilesetResolver.forGenAiTasks();
      fileset.wasmLoaderPath.h = `${msg.genaiBase}genai_wasm_internal.js`;
      fileset.wasmBinaryPath.h = msg.wasmUrl;
      const stream = await modelStream(msg);
      const t0 = performance.now();
      llm = await genai.LlmInference.createFromOptions(fileset, {
        baseOptions: { modelAssetBuffer: stream.getReader() },
        maxTokens: 2048,
        topK: 1,
        temperature: 0,
        randomSeed: 1,
        maxNumImages: 1,
      });
      post({ type: 'ready', loadSeconds: (performance.now() - t0) / 1000 });
    } catch (err) {
      post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
    }
  } else if (msg.type === 'generate') {
    if (!llm) { post({ type: 'error', id: msg.id, message: 'model not loaded' }); return; }
    const query: PromptPart[] = msg.image
      ? ['<|turn>user\n', { imageSource: msg.image }, `${msg.prompt}<turn|>\n<|turn>model\n`]
      : [`<|turn>user\n${msg.prompt}<turn|>\n<|turn>model\n`];
    let raw = '';
    try {
      await llm.generateResponse(query, (partial) => {
        raw += partial;
        post({ type: 'partial', id: msg.id, text: raw });
      });
      post({ type: 'done', id: msg.id, text: raw });
    } catch (err) {
      // cancelProcessing() rejects the pending generation: report what we have.
      post({ type: 'done', id: msg.id, text: raw });
      if (!/cancel/i.test(String(err))) console.warn('gemma generate', err);
    } finally {
      msg.image?.close();
    }
  } else if (msg.type === 'cancel') {
    llm?.cancelProcessing();
  }
};
