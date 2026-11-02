// Copies MediaPipe wasm runtime and downloads model files into public/ so the
// demo runs fully from localhost (no CDN at runtime).
import { mkdir, cp, writeFile, access } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const here = dirname(fileURLToPath(import.meta.url));
const pub = join(here, '..', 'public');
const require = createRequire(import.meta.url);
const mpRoot = join(here, '..', 'node_modules', '@mediapipe', 'tasks-vision');

const MODELS = {
  'face_landmarker.task':
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
  'selfie_multiclass_256x256.tflite':
    'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite',
};

await mkdir(join(pub, 'mediapipe', 'wasm'), { recursive: true });
await cp(join(mpRoot, 'wasm'), join(pub, 'mediapipe', 'wasm'), { recursive: true });
console.log('copied mediapipe wasm');

const litertRoot = join(here, '..', 'node_modules', '@litertjs', 'core');
await mkdir(join(pub, 'litert', 'wasm'), { recursive: true });
await cp(join(litertRoot, 'wasm'), join(pub, 'litert', 'wasm'), { recursive: true });
console.log('copied litert.js wasm');

await mkdir(join(pub, 'models'), { recursive: true });
for (const [name, url] of Object.entries(MODELS)) {
  const dst = join(pub, 'models', name);
  try {
    await access(dst);
    console.log(`exists: ${name}`);
    continue;
  } catch {}
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed ${url}: ${res.status}`);
  await writeFile(dst, Buffer.from(await res.arrayBuffer()));
  console.log(`downloaded ${name}`);
}
