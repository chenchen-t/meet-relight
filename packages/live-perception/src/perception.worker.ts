/// <reference lib="webworker" />
/**
 * MediaPipe perception worker. Runs FaceLandmarker + selfie_multiclass segmenter
 * off the main thread so the 60fps render loop never blocks on inference or the
 * GPU→CPU mask readback.
 *
 * MediaPipe's wasm loader calls importScripts(), which throws in module workers
 * (what Vite emits). We polyfill it with sync XHR + indirect eval — allowed in workers.
 */
import { FilesetResolver, FaceLandmarker, ImageSegmenter } from '@mediapipe/tasks-vision';
import type { WorkerRequest, WorkerResponse } from './protocol.ts';

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

let face: FaceLandmarker | null = null;
let seg: ImageSegmenter | null = null;
let rgba = new Uint8Array(0);

const post = (msg: WorkerResponse, transfer: Transferable[] = []) => self.postMessage(msg, transfer);

async function init(base: string): Promise<void> {
  const fileset = await FilesetResolver.forVisionTasks(`${base}mediapipe/wasm`);
  const canvas = new OffscreenCanvas(1, 1);
  [face, seg] = await Promise.all([
    FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: `${base}models/face_landmarker.task`, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFacialTransformationMatrixes: true,
      canvas,
    }),
    ImageSegmenter.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: `${base}models/selfie_multiclass_256x256.tflite`, delegate: 'GPU' },
      runningMode: 'VIDEO',
      outputCategoryMask: true,
      outputConfidenceMasks: false,
      canvas: new OffscreenCanvas(1, 1),
    }),
  ]);
}

function process(id: number, bitmap: ImageBitmap, ts: number, segment: boolean): void {
  const t0 = performance.now();
  const fr = face!.detectForVideo(bitmap, ts);
  const t1 = performance.now();
  let landmarks: Float32Array | null = null;
  let matrix: Float32Array | null = null;
  const lm = fr.faceLandmarks[0];
  if (lm) {
    landmarks = new Float32Array(lm.length * 3);
    for (let i = 0; i < lm.length; i++) {
      landmarks[i * 3] = lm[i]!.x;
      landmarks[i * 3 + 1] = lm[i]!.y;
      landmarks[i * 3 + 2] = lm[i]!.z;
    }
    const m = fr.facialTransformationMatrixes?.[0];
    if (m) matrix = Float32Array.from(m.data);
  }

  let mask: Uint8Array | null = null;
  let w = 0;
  let h = 0;
  const sr = segment ? seg!.segmentForVideo(bitmap, ts) : null;
  const cm = sr?.categoryMask;
  if (cm) {
    w = cm.width;
    h = cm.height;
    const cat = cm.getAsUint8Array();
    rgba = new Uint8Array(w * h * 4); // fresh buffer: it is transferred to the main thread
    // r = person, g = skin (face+body), b = face skin, a = hair
    for (let i = 0, j = 0; i < cat.length; i++, j += 4) {
      const c = cat[i]!;
      rgba[j] = c !== 0 ? 255 : 0;
      rgba[j + 1] = c === 2 || c === 3 ? 255 : 0;
      rgba[j + 2] = c === 3 ? 255 : 0;
      rgba[j + 3] = c === 1 ? 255 : 0;
    }
    mask = rgba;
  }
  sr?.close();
  bitmap.close();
  const t2 = performance.now();

  const transfer: Transferable[] = [];
  if (landmarks) transfer.push(landmarks.buffer);
  if (matrix) transfer.push(matrix.buffer);
  if (mask) transfer.push(mask.buffer);
  post({ type: 'result', id, landmarks, matrix, mask, maskWidth: w, maskHeight: h, faceMs: t1 - t0, segMs: t2 - t1 }, transfer);
}

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      await init(msg.base);
      post({ type: 'ready' });
    } else if (msg.type === 'frame') {
      process(msg.id, msg.bitmap, msg.timestampMs, msg.segment);
    }
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? `${err.message}\n${err.stack}` : String(err) });
  }
};
