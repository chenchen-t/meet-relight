import { initGpu, configureCanvas, GpuTimer, FrameMeter, RollingAvg } from '@relight/gpu-core';
import { EXAMPLE_PROFILE, validateTargetProfile } from '@relight/shared-schema';
import { Perception, type PerceptionFrame } from './perception.ts';
import shaderSrc from './probe.wgsl?raw';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const hud = $<HTMLDivElement>('hud');
const errEl = $<HTMLDivElement>('err');
const canvas = $<HTMLCanvasElement>('gfx');
const video = $<HTMLVideoElement>('cam');
const showMask = $<HTMLInputElement>('showMask');
const showLandmarks = $<HTMLInputElement>('showLandmarks');
const runMp = $<HTMLInputElement>('runMp');

const MAX_POINTS = 478;

async function main(): Promise<void> {
  const schemaCheck = validateTargetProfile(EXAMPLE_PROFILE);
  const gpu = await initGpu();
  const { device } = gpu;

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 60 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const { context, format } = configureCanvas(device, canvas);

  hud.textContent = 'loading MediaPipe…';
  const perception = await Perception.create();

  const module = device.createShaderModule({ code: shaderSrc });
  const fullPipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vsFull' },
    fragment: { module, entryPoint: 'fsFull', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const pointPipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vsPoint' },
    fragment: { module, entryPoint: 'fsPoint', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });

  const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const uniformBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const pointBuf = device.createBuffer({
    size: MAX_POINTS * 3 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  let maskTex = device.createTexture({
    size: [1, 1],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  const pointBG = device.createBindGroup({
    layout: pointPipe.getBindGroupLayout(0),
    entries: [
      { binding: 3, resource: { buffer: uniformBuf } },
      { binding: 4, resource: { buffer: pointBuf } },
    ],
  });

  const timer = new GpuTimer(device, gpu.hasTimestamps);
  const meter = new FrameMeter();
  const avg = {
    down: new RollingAvg(),
    face: new RollingAvg(),
    seg: new RollingAvg(),
    read: new RollingAvg(),
    upload: new RollingAvg(),
    cpuFrame: new RollingAvg(),
  };
  let numPoints = 0;
  let lastVideoTime = -1;
  let last: PerceptionFrame | null = null;

  const frame = () => {
    const tf0 = performance.now();
    meter.tick();

    if (runMp.checked && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      last = perception.process(video, performance.now());
      avg.down.push(last.timings.downscaleMs);
      avg.face.push(last.timings.faceMs);
      avg.seg.push(last.timings.segMs);
      avg.read.push(last.timings.maskReadMs);

      const tu0 = performance.now();
      if (last.landmarks) {
        numPoints = Math.min(MAX_POINTS, last.landmarks.length / 3);
        device.queue.writeBuffer(pointBuf, 0, last.landmarks, 0, numPoints * 3);
      } else numPoints = 0;
      if (last.mask) {
        if (maskTex.width !== last.maskWidth || maskTex.height !== last.maskHeight) {
          maskTex.destroy();
          maskTex = device.createTexture({
            size: [last.maskWidth, last.maskHeight],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
          });
        }
        device.queue.writeTexture(
          { texture: maskTex },
          last.mask as Uint8Array<ArrayBuffer>,
          { bytesPerRow: last.maskWidth * 4 },
          [last.maskWidth, last.maskHeight],
        );
      }
      avg.upload.push(performance.now() - tu0);
    }

    device.queue.writeBuffer(
      uniformBuf,
      0,
      new Float32Array([canvas.width, canvas.height, 1, showMask.checked ? 1 : 0, 4, numPoints, 0, 0]),
    );

    // External textures expire every frame → bind group must be recreated per frame.
    const fullBG = device.createBindGroup({
      layout: fullPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: device.importExternalTexture({ source: video }) },
        { binding: 2, resource: maskTex.createView() },
        { binding: 3, resource: { buffer: uniformBuf } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        { view: context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
      ],
      ...timer.timestampWrites(),
    });
    pass.setPipeline(fullPipe);
    pass.setBindGroup(0, fullBG);
    pass.draw(3);
    if (showLandmarks.checked && numPoints > 0) {
      pass.setPipeline(pointPipe);
      pass.setBindGroup(0, pointBG);
      pass.draw(6, numPoints);
    }
    pass.end();
    const after = timer.resolve(encoder);
    device.queue.submit([encoder.finish()]);
    after();
    avg.cpuFrame.push(performance.now() - tf0);

    hud.textContent = [
      `M0 WebGPU probe   ${canvas.width}x${canvas.height}`,
      `fps                ${meter.fps.toFixed(1)}`,
      `gpu render pass    ${timer.enabled ? timer.avgMs.toFixed(3) + ' ms' : 'n/a (no timestamp-query)'}`,
      `mp downscale (2d)  ${avg.down.value.toFixed(2)} ms`,
      `mp face landmarks  ${avg.face.value.toFixed(2)} ms`,
      `mp segmentation    ${avg.seg.value.toFixed(2)} ms`,
      `mask readback+pack ${avg.read.value.toFixed(2)} ms  (${last?.maskWidth ?? 0}x${last?.maskHeight ?? 0})`,
      `webgpu upload      ${avg.upload.value.toFixed(3)} ms`,
      `cpu frame total    ${avg.cpuFrame.value.toFixed(2)} ms`,
      `face found         ${numPoints > 0 ? 'yes (' + numPoints + ' pts)' : 'no'}`,
      `schema example     ${schemaCheck.ok ? 'valid' : 'INVALID'}`,
    ].join('\n');

    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

main().catch((e: unknown) => {
  console.error(e);
  errEl.textContent = String(e instanceof Error ? e.stack ?? e.message : e);
});
