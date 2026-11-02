import { initGpu, configureCanvas, GpuTimer, FrameMeter } from '@relight/gpu-core';
import {
  azimuthToClock,
  validateTargetProfile,
  type TargetProfile,
  type Light,
} from '@relight/shared-schema';
import {
  RelightRenderer,
  DEFAULT_PARAMS,
  ViewMode,
  ROOM_PRESETS,
  roomSH,
  shToVec4Array,
  kelvinToRgb,
  MATCAP_SIZE,
  linearToSrgb,
  type RoomLight,
  type SH9,
} from '@relight/relight-shader';
import synthSrc from './synth.wgsl?raw';
import { shEval, fibonacciSphere } from '@relight/relight-shader';
import { SampleGatherer, fitRoomLight, exposureNormalizedRoom } from '@relight/live-perception';

// ---------------------------------------------------------------- tiny UI kit
const panel = document.getElementById('panel')!;
const hud = document.getElementById('hud')!;
const errEl = document.getElementById('err')!;
const canvas = document.getElementById('gfx') as HTMLCanvasElement;
let dirty = { bake: true, synth: true, params: true };
const onAny = (what: Partial<typeof dirty>) => () => (dirty = { ...dirty, ...what });

function group(title: string): HTMLFieldSetElement {
  const fs = document.createElement('fieldset');
  fs.innerHTML = `<legend>${title}</legend>`;
  panel.appendChild(fs);
  return fs;
}
function slider(
  parent: HTMLElement, label: string, min: number, max: number, step: number, value: number,
  what: Partial<typeof dirty> = { bake: true, params: true },
): () => number {
  const row = document.createElement('div');
  row.className = 'row';
  row.innerHTML = `<span>${label}</span><input type="range" min="${min}" max="${max}" step="${step}" value="${value}"><output>${value}</output>`;
  const input = row.querySelector('input')!;
  const out = row.querySelector('output')!;
  input.addEventListener('input', () => {
    out.textContent = input.value;
    onAny(what)();
  });
  parent.appendChild(row);
  return () => Number(input.value);
}
function select<T extends string>(
  parent: HTMLElement, label: string, options: Record<T, string>, value: NoInfer<T>, what: Partial<typeof dirty>,
): () => T {
  const row = document.createElement('div');
  row.className = 'row';
  row.innerHTML = `<span>${label}</span><select style="grid-column: span 2"></select>`;
  const sel = row.querySelector('select')!;
  const fromUrl = new URLSearchParams(location.search).get(label) as T | null;
  if (fromUrl && fromUrl in options) value = fromUrl as NoInfer<T>;
  for (const [k, v] of Object.entries(options)) sel.add(new Option(v as string, k, false, k === value));
  sel.addEventListener('change', onAny(what));
  parent.appendChild(row);
  return () => sel.value as T;
}
function checkbox(parent: HTMLElement, label: string, value: boolean, what: Partial<typeof dirty>): () => boolean {
  const row = document.createElement('label');
  row.style.display = 'block';
  row.innerHTML = `<input type="checkbox" ${value ? 'checked' : ''}> ${label}`;
  const input = row.querySelector('input')!;
  input.addEventListener('change', onAny(what));
  parent.appendChild(row);
  return () => input.checked;
}

// ---------------------------------------------------------------- controls
const gView = group('View');
const ui = {
  res: select(gView, 'resolution', { '1080': '1920×1080', '2160': '3840×2160 (4K)', '720': '1280×720' }, '1080', { synth: true }),
  mode: select(
    gView, 'mode',
    { '2': 'split (orig | relit)', '0': 'relit', '1': 'original', '3': 'normals', '4': 'quotient ratio', '5': 'mask' },
    '2', { params: true },
  ),
  split: slider(gView, 'split x', 0, 1, 0.01, 0.5, { params: true }),
  strength: slider(gView, 'strength', 0, 1, 0.01, 1, { params: true }),
};
const gRoom = group('Room light (what the camera sees)');
const room = {
  preset: select(
    gRoom, 'preset',
    { overhead: 'Overhead ceiling panel', window_left: 'Window (viewer left)', desk_lamp: 'Desk lamp + screen', flat: 'Flat' },
    'overhead', { synth: true, bake: true },
  ),
  known: checkbox(gRoom, 'room SH known (de-light) — off = flat assumption', true, { bake: true }),
  useFit: checkbox(gRoom, 'use Greenroom-fitted SH instead of ground truth', new URLSearchParams(location.search).has('autofit'), { bake: true }),
};
const fitBtn = document.createElement('button');
fitBtn.textContent = 'Greenroom: fit room SH from rendered image';
fitBtn.style.cssText = 'width:100%;padding:5px;margin:4px 0;background:#2a4d7a;color:#fff;border:0;border-radius:4px;cursor:pointer';
gRoom.appendChild(fitBtn);
const fitStats = document.createElement('div');
fitStats.style.cssText = 'font-family:ui-monospace,monospace;font-size:11px;color:#aaa;white-space:pre-wrap';
gRoom.appendChild(fitStats);
const gKey = group('Key light');
const key = {
  az: slider(gKey, 'azimuth°', -180, 180, 1, -45),
  el: slider(gKey, 'elevation°', -60, 90, 1, 30),
  intensity: slider(gKey, 'intensity', 0, 2, 0.01, 0.85),
  softness: slider(gKey, 'softness', 0, 1, 0.01, 0.75),
  kelvin: slider(gKey, 'kelvin', 2500, 9000, 50, 5200),
};
const gFill = group('Fill light');
const fill = {
  az: slider(gFill, 'azimuth°', -180, 180, 1, 50),
  el: slider(gFill, 'elevation°', -60, 90, 1, 10),
  intensity: slider(gFill, 'intensity', 0, 2, 0.01, 0.27),
  softness: slider(gFill, 'softness', 0, 1, 0.01, 1),
};
const gRim = group('Rim light (Fresnel)');
const rim = {
  intensity: slider(gRim, 'intensity', 0, 2, 0.01, 0.35),
  kelvin: slider(gRim, 'kelvin', 2500, 12000, 50, 8000),
};
const gMat = group('Ambient & material');
const mat = {
  ambient: slider(gMat, 'ambient', 0, 0.6, 0.01, 0.15),
  roughness: slider(gMat, 'roughness', 0.05, 1, 0.01, 0.5, { params: true }),
  spec: slider(gMat, 'spec strength', 0, 1, 0.01, 0.15, { params: true }),
};
const gOut = group('Quotient MatCap (64×64) & TargetProfile');
const matcapCanvas = document.createElement('canvas');
matcapCanvas.id = 'matcap';
matcapCanvas.width = matcapCanvas.height = MATCAP_SIZE;
gOut.appendChild(matcapCanvas);
const validity = document.createElement('div');
gOut.appendChild(validity);
const jsonPre = document.createElement('pre');
jsonPre.id = 'json';
gOut.appendChild(jsonPre);

// ---------------------------------------------------------------- profile from UI
function buildProfile(): TargetProfile {
  const mk = (id: string, role: Light['role'], az: number, el: number, intensity: number, softness: number, kelvin: number, shape: Light['shape']): Light => ({
    id, role, azimuth_deg: az, elevation_deg: el, clock_direction: azimuthToClock(az), intensity,
    color_rgb: kelvinToRgb(kelvin).map((v) => +v.toFixed(3)) as [number, number, number],
    kelvin, softness, shape, assigned_execution: 'virtual_shader',
  });
  const a = mat.ambient();
  return {
    subject_type: 'portrait',
    lighting_rig: {
      ambient_rgb: [a, a, a * 1.05],
      key_to_fill_ratio: +(key.intensity() / Math.max(fill.intensity(), 1e-3)).toFixed(2),
      lights: [
        mk('key_light', 'key', key.az(), key.el(), key.intensity(), key.softness(), key.kelvin(), 'octagon'),
        mk('fill_light', 'fill', fill.az(), fill.el(), fill.intensity(), fill.softness(), 5600, 'rect_softbox'),
        mk('rim_light', 'rim', 160, 20, rim.intensity(), 0.3, rim.kelvin(), 'stripbox'),
      ],
    },
    color_dna: { lab_mean: [62.5, 4.2, 8.1], lab_std: [18.4, 6.1, 9.5] },
  };
}

function drawMatcapPreview(data: Float32Array): void {
  const ctx = matcapCanvas.getContext('2d')!;
  const img = ctx.createImageData(MATCAP_SIZE, MATCAP_SIZE);
  for (let i = 0; i < MATCAP_SIZE * MATCAP_SIZE; i++) {
    for (let c = 0; c < 3; c++) img.data[i * 4 + c] = 255 * linearToSrgb(Math.min(1, data[i * 4 + c]! * 0.5));
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// ---------------------------------------------------------------- GPU
async function main(): Promise<void> {
  const gpu = await initGpu();
  const { device } = gpu;
  const { context, format } = configureCanvas(device, canvas);
  const relight = new RelightRenderer(device, format, 'texture');

  const synthModule = device.createShaderModule({ label: 'synth', code: synthSrc });
  const synthPipe = device.createComputePipeline({ layout: 'auto', compute: { module: synthModule, entryPoint: 'main' } });
  const synthUniform = device.createBuffer({ size: 16 + 9 * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  let tex: { color: GPUTexture; geo: GPUTexture; mask: GPUTexture } | null = null;
  let roomSh: SH9 = roomSH(ROOM_PRESETS.overhead!);

  const relightTimer = new GpuTimer(device, gpu.hasTimestamps);
  const meter = new FrameMeter();
  let bakeMs = 0;

  const runSynth = () => {
    const h = Number(ui.res());
    const w = Math.round((h * 16) / 9);
    if (!tex || tex.color.height !== h) {
      tex?.color.destroy(); tex?.geo.destroy(); tex?.mask.destroy();
      const mk = (f: GPUTextureFormat) =>
        device.createTexture({ size: [w, h], format: f, usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
      tex = { color: mk('rgba8unorm'), geo: mk('rgba16float'), mask: mk('rgba8unorm') };
      canvas.width = w;
      canvas.height = h;
    }
    roomSh = roomSH(ROOM_PRESETS[room.preset()]!);
    const ub = new Float32Array(4 + 36);
    ub.set([w, h, 0, 0]);
    ub.set(shToVec4Array(roomSh), 4);
    device.queue.writeBuffer(synthUniform, 0, ub);
    const bg = device.createBindGroup({
      layout: synthPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: synthUniform } },
        { binding: 1, resource: tex.color.createView() },
        { binding: 2, resource: tex.geo.createView() },
        { binding: 3, resource: tex.mask.createView() },
      ],
    });
    const enc = device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(synthPipe);
    cp.setBindGroup(0, bg);
    cp.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
    cp.end();
    device.queue.submit([enc.finish()]);
  };

  const gatherer = new SampleGatherer(device, 'texture', 2);
  let fitted: { shRoom: SH9; samples: Float32Array } | null = null;
  const runFit = async () => {
    if (!tex) return;
    const enc = device.createCommandEncoder();
    const read = gatherer.encode(enc, tex.color.createView(), tex.geo, tex.mask.createView());
    device.queue.submit([enc.finish()]);
    const samples = await read();
    const t0 = performance.now();
    const fit = fitRoomLight(samples);
    const fitMs = performance.now() - t0;
    // Scale-free shading-shape error vs ground truth over camera-facing normals.
    const lum = (v: number[]) => 0.2126 * v[0]! + 0.7152 * v[1]! + 0.0722 * v[2]!;
    const fF = lum(shEval(fit.shRaw, [0, 0, 1]));
    const gF = lum(shEval(roomSh, [0, 0, 1]));
    let err = 0;
    let cnt = 0;
    for (const n of fibonacciSphere(600)) {
      if (n[2] < 0.4) continue;
      const g = lum(shEval(roomSh, n)) / gF;
      err += Math.abs(lum(shEval(fit.shRaw, n)) / fF - g) / Math.max(g, 0.1);
      cnt++;
    }
    fitted = { shRoom: fit.shRoom, samples };
    fitStats.textContent =
      `fit: ${fit.inlierCount}/${fit.sampleCount} inliers, ${fitMs.toFixed(1)} ms\n` +
      `residual ${(fit.rmsRel * 100).toFixed(1)}% rms, coverage ${(fit.coverage * 100).toFixed(0)}%\n` +
      `shape error vs ground truth: ${((err / cnt) * 100).toFixed(1)}%`;
    dirty.bake = true;
  };
  fitBtn.onclick = () => void runFit();
  if (new URLSearchParams(location.search).has('autofit')) setTimeout(() => void runFit(), 800);

  const frame = () => {
    meter.tick();
    if (dirty.synth) {
      runSynth();
      fitted = null;
    }
    if (dirty.bake || dirty.synth) {
      const profile = buildProfile();
      let roomModel: RoomLight = room.known() ? { kind: 'sh', sh: roomSh } : { kind: 'flat', level: 0.75 };
      if (room.known() && room.useFit() && fitted) {
        roomModel = { kind: 'sh', sh: exposureNormalizedRoom(fitted.shRoom, fitted.samples, profile.lighting_rig).sh };
      }
      bakeMs = relight.bake(profile, roomModel);
      drawMatcapPreview(relight.lastBake!);
      const v = validateTargetProfile(profile);
      validity.innerHTML = v.ok
        ? '<span class="ok">✓ TargetProfile valid against schema</span>'
        : `<span class="bad">✗ ${v.errors.join('; ')}</span>`;
      jsonPre.textContent = JSON.stringify(profile, null, 1);
      dirty.params = true;
    }
    if (dirty.params) {
      const rimRgb = kelvinToRgb(rim.kelvin()).map((c) => c * rim.intensity() * 0.6) as [number, number, number];
      relight.setParams({
        ...DEFAULT_PARAMS,
        mode: Number(ui.mode()) as ViewMode,
        splitX: ui.split(),
        strength: ui.strength(),
        roughness: mat.roughness(),
        specStrength: mat.spec(),
        rimColor: rimRgb,
      });
    }
    dirty = { bake: false, synth: false, params: false };

    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      ...relightTimer.timestampWrites(),
    });
    relight.draw(pass, { source: tex!.color.createView(), geo: tex!.geo.createView(), mask: tex!.mask.createView() });
    pass.end();
    const after = relightTimer.resolve(enc);
    device.queue.submit([enc.finish()]);
    after();

    hud.textContent = [
      `M1 relight-shader  ${canvas.width}x${canvas.height}`,
      `fps               ${meter.fps.toFixed(1)}`,
      `relight pass GPU  ${relightTimer.enabled ? relightTimer.avgMs.toFixed(3) + ' ms' : 'n/a'}`,
      `matcap bake CPU   ${bakeMs.toFixed(2)} ms`,
      `room              ${room.preset()}${room.known() ? ' (SH known)' : ' (flat assumed)'}`,
    ].join('\n');
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

main().catch((e: unknown) => {
  console.error(e);
  errEl.textContent = String(e instanceof Error ? e.stack ?? e.message : e);
});
