// Dev tool: /inspect.html?m=models/a.tflite,models/b.tflite → prints I/O details per model (WebGPU compile).
import { loadLiteRt, loadAndCompile, setWebGpuDevice } from '@litertjs/core';
import { initGpu } from '@relight/gpu-core';

const out = document.getElementById('out')!;
const models = (new URLSearchParams(location.search).get('m') ?? '').split(',').filter(Boolean);
(async () => {
  const { device } = await initGpu();
  await loadLiteRt('./litert/wasm/');
  setWebGpuDevice(device);
  const res: Record<string, unknown> = {};
  for (const m of models) {
    try {
      const cm = await loadAndCompile(`/${m}`, { accelerator: 'webgpu' });
      const fmt = (d: { name: string; dtype: string; shape: Int32Array }) => `${d.name} ${d.dtype}[${[...d.shape]}]`;
      res[m] = { inputs: cm.getInputDetails().map(fmt), outputs: cm.getOutputDetails().map(fmt), signatures: Object.keys(cm.signatures) };
    } catch (e) {
      res[m] = { error: String(e) };
    }
  }
  out.textContent = JSON.stringify(res, null, 2);
  out.dataset.done = '1';
})();
