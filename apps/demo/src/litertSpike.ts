import { initGpu, configureCanvas, FrameMeter, RollingAvg } from '@relight/gpu-core';
import { LiteRtSegmenter } from '@relight/live-perception';
import probeSrc from './probe.wgsl?raw';

const hud = document.getElementById('hud')!;
const errEl = document.getElementById('err')!;
const canvas = document.getElementById('gfx') as HTMLCanvasElement;
const video = document.getElementById('cam') as HTMLVideoElement;

// Capture console warnings from LiteRT.js (e.g. WebGPU → WASM fallback notices).
const litertLogs: string[] = [];
for (const k of ['warn', 'info', 'log'] as const) {
  const orig = console[k].bind(console);
  console[k] = (...a: unknown[]) => {
    const s = a.map(String).join(' ');
    if (/litert|delegat|fallback|webgpu|wasm/i.test(s)) litertLogs.push(`[${k}] ${s.slice(0, 160)}`);
    orig(...a);
  };
}

async function main(): Promise<void> {
  const gpu = await initGpu();
  const { device } = gpu;
  device.addEventListener('uncapturederror', (e) => (errEl.textContent += `\n[gpu] ${(e as GPUUncapturedErrorEvent).error.message}`));
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const { context, format } = configureCanvas(device, canvas);

  hud.textContent = 'loading LiteRT.js + models…';
  const tLoad = performance.now();
  const seg = await LiteRtSegmenter.create(device);
  const loadMs = performance.now() - tLoad;

  // Reuse the M0 probe shader to overlay the mask (fg blue, skin orange, hair green).
  const module = device.createShaderModule({ code: probeSrc });
  const pipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vsFull' },
    fragment: { module, entryPoint: 'fsFull', targets: [{ format }] },
  });
  const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const ubuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(ubuf, 0, new Float32Array([canvas.width, canvas.height, 1, 1, 0, 0, 0, 0]));

  const meter = new FrameMeter();
  const segMeter = new FrameMeter();
  const issue = new RollingAvg(0.1);
  const gpuMs = new RollingAvg(0.1);
  let runs = 0;

  const frame = () => {
    meter.tick();
    const ext = device.importExternalTexture({ source: video });
    if (!seg.busy) {
      void seg.run(ext).then(() => {
        runs++;
        segMeter.tick();
        if (runs > 5) {
          issue.push(seg.timings.issueMs);
          gpuMs.push(seg.timings.gpuMs);
        }
      }, (e) => (errEl.textContent = String(e?.stack ?? e)));
    }
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    pass.setPipeline(pipe);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: ext },
        { binding: 2, resource: seg.mask.createView() },
        { binding: 3, resource: { buffer: ubuf } },
      ],
    }));
    pass.draw(3);
    pass.end();
    device.queue.submit([enc.finish()]);

    hud.textContent = [
      `LiteRT.js WebGPU segmentation   camera ${canvas.width}x${canvas.height}  → frame ${seg.frameW}x${seg.frameH}`,
      `shared GPUDevice  ${gpu.adapter.info?.vendor ?? ''} ${gpu.adapter.info?.architecture ?? ''}`,
      `load + compile    ${loadMs.toFixed(0)} ms`,
      `render fps        ${meter.fps.toFixed(1)}`,
      `segmentation fps  ${segMeter.fps.toFixed(1)}   (runs ${runs})`,
      `issue (3 runs)    ${issue.value.toFixed(2)} ms   CPU`,
      `chain GPU latency ${gpuMs.value.toFixed(2)} ms   pack → pre → model → post → unpack`,
      `CPU readback      none`,
      '',
      seg.describe(),
      '',
      ...litertLogs.slice(-6),
    ].join('\n');
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

main().catch((e: unknown) => {
  console.error(e);
  errEl.textContent = String(e instanceof Error ? e.stack ?? e.message : e);
});
