import { initGpu, configureCanvas, GpuTimer, FrameMeter, RollingAvg } from '@relight/gpu-core';
import { lightDirection } from '@relight/shared-schema';
import { RelightRenderer, DEFAULT_PARAMS, ViewMode, MATCAP_SIZE, type RoomLight } from '@relight/relight-shader';
import {
  PerceptionClient,
  GeoBuilder,
  OneEuroArray,
  buildFaceVertices,
  expandFaceSilhouette,
  estimateFaceExpandProfile,
  medianProfile,
  headPoseFromLandmarks,
  headProxyFromLandmarks,
  shadowLayerWeights,
  GreenroomController,
  LiteRtFace,
  LiteRtSegmenter,
  MatteRefiner,
  ShadowBaker,
  exposureNormalizedRoom,
  FACE_VERTEX_COUNT,
  type PerceptionResult,
  type HeadPose,
  type HeadProxy,
} from '@relight/live-perception';
import { createPanel, createRigControls, drawMatcapPreview, LIGHTING_PRESETS, PANEL_CSS } from './ui.ts';
import { LightDome } from './lightDome.ts';
import { estimateLightingFromBackground } from './bgLight.ts';
import { createGemmaPanel } from './gemma/panel.ts';
import { applyLayout } from './layout.ts';
import type { LightingTarget } from './gemma/agent.ts';

document.head.insertAdjacentHTML('beforeend', `<style>${PANEL_CSS} button { width: 100%; padding: 6px; margin: 4px 0; background: #2a4d7a; color: #fff; border: 0; border-radius: 4px; cursor: pointer; } #gr { font-family: ui-monospace, monospace; font-size: 11px; color: #aaa; white-space: pre-wrap; } #banner { position: absolute; bottom: 18px; left: 50%; transform: translateX(-50%); background: rgba(20,60,120,.85); padding: 10px 18px; border-radius: 8px; font-size: 15px; display: none; } .light-dome { position: absolute; right: 12px; bottom: 12px; touch-action: none; user-select: none; }</style>`);
const hud = document.getElementById('hud')!;
const errEl = document.getElementById('err')!;
const canvas = document.getElementById('gfx') as HTMLCanvasElement;
const video = document.getElementById('cam') as HTMLVideoElement;
const banner = document.createElement('div');
banner.id = 'banner';
document.getElementById('view')!.appendChild(banner);

const dirty = new Set<string>(['bake', 'params', 'shadow']);
const p = createPanel(document.getElementById('panel')!, (w) => {
  if (w.includes('camera')) {
    const url = new URL(location.href);
    url.searchParams.set('camera', ui.camera());
    location.href = url.toString();
    return;
  }
  w.forEach((x) => dirty.add(x));
});

const gView = p.group('View');
const ui = {
  mode: p.select(gView, 'mode', { '2': 'split (orig | relit)', '0': 'relit', '1': 'original', '3': 'normals', '4': 'quotient ratio', '5': 'mask', '6': 'cast shadow' }, '2', ['params']),
  split: p.slider(gView, 'split x', 0, 1, 0.01, 0.5, ['params']),
  strength: p.slider(gView, 'strength', 0, 1, 0.01, 0.5, ['params']),
  mirror: p.checkbox(gView, 'mirror (selfie view)', true, ['params']),
  dome: p.checkbox(gView, 'show light dome (drag lights)', true, ['params']),
  camera: p.select(
    gView, 'camera',
    { '1080p': '1080p (ideal 60 fps, usually 30)', '720p60': '720p @ 60 fps (lower latency if supported)' },
    '1080p', ['camera'],
  ),
  segBackend: p.select(
    gView, 'segmentation',
    { litert: 'LiteRT.js WebGPU (Tensor API graphs, zero readback)', mediapipe: 'MediaPipe (WebGL worker + CPU readback)' },
    'litert', [],
  ),
  faceBackend: p.select(
    gView, 'face',
    { litert: 'LiteRT.js WebGPU (Tensor API graphs, 5.7 KB readback)', mediapipe: 'MediaPipe (WebGL worker)' },
    'litert', [],
  ),
  latency: p.select(
    gView, 'latency',
    {
      frame: 'same-frame (Meet-style): wait for inference, zero misalignment',
      predict: 'low latency: live video + predicted geometry',
      sync: 'frame-synced: exact alignment, delayed video (MediaPipe worker)',
    },
    'frame', [],
  ),
};

const gGreen = p.group('Greenroom (de-light)');
const startBtn = document.createElement('button');
startBtn.textContent = '▶ Start Greenroom (turn head slowly ↔ for 1.5 s)';
gGreen.appendChild(startBtn);
const green = {
  auto: p.checkbox(gGreen, 'auto-start when a face appears', true, []),
  delight: p.checkbox(gGreen, 'de-light with fitted room SH (off = flat / add-light only)', true, ['bake']),
  heartbeat: p.checkbox(gGreen, 'heartbeat refit (5 s / >15% jump) + 1 s cross-fade', true, []),
  exposure: p.slider(gGreen, 'exposure', 0.4, 2, 0.01, 1),
  roomLevel: p.slider(gGreen, 'flat level', 0.2, 1.5, 0.01, 0.75),
};
const grStats = document.createElement('div');
grStats.id = 'gr';
gGreen.appendChild(grStats);

const gShadow = p.group('Cast shadow (canonical UV, key light)');
const shadowOn = p.checkbox(gShadow, 'shadows enabled', true, ['params']);
const shadowSamples = p.slider(gShadow, 'ray samples', 1, 64, 1, 16, ['shadow']);
const shadowStrength = p.slider(gShadow, 'strength', 0, 1, 0.01, 0.85, ['params']);
const gEdge = p.group('Silhouette & hair');
const edgeFade = p.slider(gEdge, 'edge fade', 0, 1, 0.01, 0.6, ['params']);
const edgeSoft = p.slider(gEdge, 'edge softness (px)', 0, 24, 1, 8, []);
const hlSoft = p.slider(gEdge, 'highlight softness', 0, 1, 0.05, 0.6, ['params']);
const nSmooth = p.slider(gEdge, 'normal smoothing (px)', 0, 12, 1, 5, []);
const relightHair = p.checkbox(gEdge, 'relight hair', true, []);

const gPer = p.group('Perception');
const per = {
  minCutoff: p.slider(gPer, 'smooth cutoff', 0.1, 5, 0.1, 1.5, []),
  beta: p.slider(gPer, 'smooth beta', 0, 60, 0.5, 20, []),
  predictGain: p.slider(gPer, 'predict gain', 0, 1.5, 0.05, 1, []),
  blob: p.slider(gPer, 'body normal', 0, 40, 1, 18, []),
  feather: p.slider(gPer, 'face feather', 2, 30, 1, 14, []),
  flatten: p.slider(gPer, 'head flatten', 0.6, 2, 0.05, 1.25, []),
  expand: p.slider(gPer, 'face mesh expand', 0, 0.2, 0.01, 0.08, []),
  autoFit: p.checkbox(gPer, 'auto-fit mesh boundary to my face (Greenroom)', true, []),
  snap: p.checkbox(gPer, 'snap mesh to current frame (affine; same-frame mode)', true, []),
};
const gPreset = p.group('Lighting presets');
const rig = createRigControls(p, ['bake', 'params', 'shadow']);
{
  const grid = document.createElement('div');
  grid.style.cssText = 'display:grid;grid-template-columns:1fr 1fr;gap:4px';
  const buttons: HTMLButtonElement[] = [];
  for (const pr of LIGHTING_PRESETS) {
    const b = document.createElement('button');
    b.textContent = pr.name;
    b.style.cssText = 'margin:0;padding:5px 4px;font-size:12px;background:#2c2c2c';
    b.onclick = () => {
      rig.applyPreset(pr);
      for (const o of buttons) o.style.background = o === b ? '#2a4d7a' : '#2c2c2c';
    };
    buttons.push(b);
    grid.appendChild(b);
  }
  gPreset.appendChild(grid);
  gView.after(gPreset);
}

// Absolute asset base (works under any hosting sub-path; workers resolve relative URLs differently).
const ASSET_BASE = new URL(import.meta.env.BASE_URL, location.href).href;
// Virtual backgrounds, each paired with a lighting preset whose key comes from the scene's light side.
const SCENES: { name: string; url: string | null; preset?: string }[] = [
  { name: '📷 Camera', url: null },
  { name: 'Beach', url: `${ASSET_BASE}backgrounds/beach.jpg`, preset: 'Beach midday' },
  { name: 'Sunset sea', url: `${ASSET_BASE}backgrounds/sunset_sea.jpg`, preset: 'Golden hour' },
  { name: 'Cozy room', url: `${ASSET_BASE}backgrounds/cozy_room.jpg`, preset: 'Warm lamp' },
  { name: 'Neon city', url: `${ASSET_BASE}backgrounds/neon_city.jpg`, preset: 'Cyberpunk' },
  { name: 'Office', url: `${ASSET_BASE}backgrounds/office.jpg`, preset: 'Soft beauty' },
  { name: 'Halloween', url: `${ASSET_BASE}backgrounds/halloween.jpg`, preset: 'Horror' },
];
const gBg = p.group('Background');
const bgState = { image: null as ImageBitmap | null, dirty: false, hairBefore: null as boolean | null };
const bgMatch = p.select(
  gBg, 'lighting',
  { smart: 'match: curated preset, else auto-estimate', auto: 'match: auto-estimate from image', off: 'keep my lighting' },
  'smart', [],
);
const bgWrap = p.slider(gBg, 'light wrap', 0, 1, 0.05, 0.35, ['params']);
const bgRefine = p.checkbox(gBg, 'refine edges (guided filter: hair, fingers)', true, []);
const bgTemporal = p.checkbox(gBg, 'temporal edge stabilisation', true, []);
const bgInfo = document.createElement('div');
bgInfo.style.cssText = 'font: 11px ui-monospace, monospace; color: #9ab; white-space: pre-wrap; margin-top: 4px';

/** Switch background (null = camera) and match the lighting to it. */
async function selectBackground(image: ImageBitmap | null, presetName?: string): Promise<void> {
  bgState.image = image;
  bgState.dirty = true;
  // With a virtual background the hair must take the scene's light too, or it reads as a cut-out.
  if (image) {
    if (bgState.hairBefore === null) bgState.hairBefore = relightHair();
    relightHair.set(true);
  } else if (bgState.hairBefore !== null) {
    relightHair.set(bgState.hairBefore);
    bgState.hairBefore = null;
  }
  bgInfo.textContent = '';
  if (!image) {
    // Back to the real camera = back to the default look: neutral white Studio rig, default strength.
    const studio = LIGHTING_PRESETS.find((x) => x.name.includes('Studio'));
    if (studio) rig.applyPreset(studio);
    ui.strength.set(0.5);
    bgInfo.textContent = 'lighting: default (Studio, white)';
    return;
  }
  if (bgMatch() === 'off') return;
  const curated = presetName ? LIGHTING_PRESETS.find((x) => x.name.includes(presetName)) : undefined;
  if (curated && bgMatch() === 'smart') {
    // Curated scene lights are on the screen-left; presets put the key screen-right in the mirrored view.
    rig.applyPreset(curated, ui.mirror() ? -1 : 1);
    bgInfo.textContent = `lighting: ${curated.name} (curated)`;
    return;
  }
  const est = estimateLightingFromBackground(image, ui.mirror());
  rig.applyPreset(est.preset);
  const k = est.preset.key;
  bgInfo.textContent =
    `lighting: auto from image\n key ${k.az}° az / ${k.el}° el, ${k.gel}, soft ${k.softness}\n` +
    ` source at x=${est.keyUv[0].toFixed(2)} y=${est.keyUv[1].toFixed(2)}, contrast ${est.contrast.toFixed(1)}`;
}
{
  const grid = document.createElement('div');
  grid.style.cssText = 'display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin-bottom:6px';
  const buttons: HTMLButtonElement[] = [];
  const highlight = (b: HTMLButtonElement) => { for (const o of buttons) o.style.borderColor = o === b ? '#4c8dff' : 'transparent'; };
  const style = (url: string | null) => `margin:0;padding:0;height:46px;font-size:11px;font-weight:600;text-shadow:0 1px 3px #000;border:2px solid transparent;` +
    (url ? `background:#222 url(${url}) center/cover` : 'background:#2c2c2c');
  for (const sc of SCENES) {
    const b = document.createElement('button');
    b.textContent = sc.name;
    b.style.cssText = style(sc.url);
    b.onclick = async () => {
      highlight(b);
      const img = sc.url ? await createImageBitmap(await (await fetch(sc.url)).blob()) : null;
      await selectBackground(img, sc.preset);
    };
    buttons.push(b);
    grid.appendChild(b);
  }
  // Upload your own background → lighting is auto-estimated from it.
  const file = document.createElement('input');
  file.type = 'file';
  file.accept = 'image/*';
  file.style.display = 'none';
  const up = document.createElement('button');
  up.textContent = '＋ Upload';
  up.style.cssText = style(null);
  up.onclick = () => file.click();
  file.onchange = async () => {
    const f = file.files?.[0];
    if (!f) return;
    const url = URL.createObjectURL(f);
    up.style.cssText = style(url);
    up.textContent = 'Custom';
    highlight(up);
    await selectBackground(await createImageBitmap(f));
  };
  buttons.push(up);
  grid.appendChild(up);
  grid.appendChild(file);
  buttons[0]!.style.borderColor = '#4c8dff';
  gBg.insertBefore(grid, gBg.children[1]!);
  gBg.appendChild(bgInfo);
  gView.after(gBg);
}

// Gemma 4 (on-device, WebGPU): words / voice / reference photo → lighting tool calls.
{
  const gGemma = p.group('✨ Gemma 4 lighting assistant');
  const plain = (s: string) => s.replace(/^[^\p{L}\p{N}]+/u, '').trim();
  const target: LightingTarget = {
    getPreset: () => rig.getPreset(),
    applyPreset: (pr) => rig.applyPreset(pr),
    mirrored: () => ui.mirror(),
    presetNames: () => LIGHTING_PRESETS.map((x) => plain(x.name)),
    applyPresetByName: (name) => {
      const pr = LIGHTING_PRESETS.find((x) => plain(x.name).toLowerCase() === plain(name).toLowerCase()) ??
        LIGHTING_PRESETS.find((x) => x.name.toLowerCase().includes(plain(name).toLowerCase()));
      if (pr) rig.applyPreset(pr);
      return !!pr;
    },
    backgroundNames: () => SCENES.map((s) => plain(s.name)),
    setBackground: async (name) => {
      const sc = SCENES.find((s) => plain(s.name).toLowerCase() === plain(name).toLowerCase());
      if (!sc) return false;
      const img = sc.url ? await createImageBitmap(await (await fetch(sc.url)).blob()) : null;
      await selectBackground(img, sc.preset);
      return true;
    },
  };
  createGemmaPanel(gGemma, target, ASSET_BASE);
  gView.after(gGemma);
}
const dome = new LightDome(document.getElementById('view')!, {
  onMove: (role, az, el) => {
    rig.positions[role].az.set(az);
    rig.positions[role].el.set(el);
  },
});
// Split view: drag the orig | relit divider directly on the video.
{
  const toSplit = (e: PointerEvent) => {
    const r = canvas.getBoundingClientRect();
    return Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
  };
  let dragging = false;
  canvas.addEventListener('pointerdown', (e) => {
    if (ui.mode() !== '2') return;
    dragging = true;
    canvas.setPointerCapture(e.pointerId);
    ui.split.set(+toSplit(e).toFixed(3));
  });
  canvas.addEventListener('pointermove', (e) => {
    if (ui.mode() !== '2') {
      canvas.style.cursor = '';
      return;
    }
    if (dragging) ui.split.set(+toSplit(e).toFixed(3));
    const near = Math.abs(toSplit(e) - ui.split()) * canvas.getBoundingClientRect().width < 14;
    canvas.style.cursor = dragging || near ? 'ew-resize' : 'pointer';
  });
  const end = () => (dragging = false);
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
}
const gOut = p.group('Quotient MatCap (64×64)');
const matcapCanvas = document.createElement('canvas');
matcapCanvas.id = 'matcap';
matcapCanvas.width = matcapCanvas.height = MATCAP_SIZE;
gOut.appendChild(matcapCanvas);

// ---- Layout: quick bar (view mode, strength) + tabs (Look / Background / Lights / Advanced).
let hudExpanded = false;
hud.title = 'click for detailed stats';
hud.onclick = () => (hudExpanded = !hudExpanded);
{
  applyLayout({
    panel: document.getElementById('panel')!,
    title: '🎬 Meet-Relight',
    subtitle: 'Powered by LiteRT.js (WebGPU) · Tensor API · MediaPipe',
    modeLabel: 'mode',
    modes: [['Before | After', '2'], ['Relit', '0'], ['Original', '1']],
    quickRows: ['strength', 'relight hair', 'mirror', 'show light dome'],
    tabs: [
      {
        id: 'look', label: '✨ Look', hint: 'Describe the look, pick a background, or choose a preset.',
        sections: [/Gemma 4/, /^Background/, /Lighting presets/],
      },
      {
        id: 'lights', label: '💡 Lights', hint: 'Drag the lights on the dome (bottom-right of the video), or fine-tune here.',
        sections: [/Key light/, /Fill light/, /Rim light/, /Ambient/, /Cast shadow/], collapsed: [/Ambient/, /Cast shadow/, /Fill light/, /Rim light/],
      },
      {
        id: 'adv', label: '⚙ Tune', hint: 'Pipeline, de-lighting and perception tuning.',
        sections: [/^View/, /Greenroom/, /Silhouette/, /Perception/, /Quotient/], collapsed: [/Silhouette/, /Perception/, /Quotient/],
      },
    ],
  });
  const viewLegend = [...document.querySelectorAll('legend')].find((l) => l.textContent === 'View');
  if (viewLegend) viewLegend.textContent = 'Pipeline & debug views';
}

const PROC_LONG_EDGE = 512;

async function main(): Promise<void> {
  const gpu = await initGpu();
  const { device } = gpu;

  const hd: MediaTrackConstraints = { width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 60 } };
  const fast: MediaTrackConstraints = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60, min: 50 } };
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: ui.camera() === '720p60' ? fast : hd, audio: false });
  } catch (e) {
    if (ui.camera() !== '720p60') throw e;
    console.warn('720p60 not supported by this camera, falling back', e);
    banner.textContent = 'This camera has no 60 fps mode — using default';
    banner.style.display = 'block';
    setTimeout(() => (banner.style.display = 'none'), 4000);
    stream = await navigator.mediaDevices.getUserMedia({ video: hd, audio: false });
  }
  video.srcObject = stream;
  await video.play();
  const W = video.videoWidth;
  const H = video.videoHeight;
  canvas.width = W;
  canvas.height = H;
  const { context, format } = configureCanvas(device, canvas);
  const s = PROC_LONG_EDGE / Math.max(W, H);
  const procW = Math.round(W * s);
  const procH = Math.round(H * s);

  // MediaPipe worker (WebGL + ~40 MB WASM/models) is only a fallback: create it lazily, the first time
  // a frame actually needs it (LiteRT unavailable, or the MediaPipe backend selected in the panel).
  let client: PerceptionClient | null = null;
  let clientLoading = false;
  hud.textContent = 'loading LiteRT.js models…';
  let seg: LiteRtSegmenter | null = null;
  try {
    seg = await LiteRtSegmenter.create(device, ASSET_BASE);
  } catch (e) {
    console.warn('LiteRT.js segmentation unavailable, falling back to MediaPipe', e);
  }
  const segMeter = new FrameMeter();
  const segGpu = new RollingAvg(0.1);
  const useLitert = () => ui.segBackend() === 'litert' && !!seg;
  let face: LiteRtFace | null = null;
  if (seg) {
    try {
      face = await LiteRtFace.create(device, ASSET_BASE);
    } catch (e) {
      console.warn('LiteRT.js face unavailable, falling back to MediaPipe', e);
    }
  }
  // LiteRT face reuses the segmenter's packed frame, so it requires the LiteRT segmentation backend.
  const useLitertFace = () => ui.faceBackend() === 'litert' && !!face && useLitert();
  const faceLm = new RollingAvg(0.1);
  const relight = new RelightRenderer(device, format, 'external');
  const geo = new GeoBuilder(device, Math.round(W / 2), Math.round(H / 2));
  const matte = new MatteRefiner(device, Math.round(W / 2), Math.round(H / 2));
  const greenroom = new GreenroomController(device, 'external');
  const shadows = new ShadowBaker(device);
  const euro = new OneEuroArray();
  greenroom.onChange = () => {
    if (fitProfiles.length) finishFit(); // end of a capture → apply the measured mesh boundary
    dirty.add('bake');
  };
  startBtn.onclick = () => { fitProfiles = []; greenroom.start(); };

  const timer = new GpuTimer(device, gpu.hasTimestamps);
  const meter = new FrameMeter();
  const perfMeter = new FrameMeter();
  const camMeter = new FrameMeter();
  const capToShow = new RollingAvg(0.1);
  const avg = { face: new RollingAvg(), seg: new RollingAvg(), lat: new RollingAvg(), cpuGeo: new RollingAvg(), bake: new RollingAvg(1) };

  const frames = new Map<number, VideoFrame>();
  const captureTimes = new Map<number, number>();
  let nextId = 1;
  let display: VideoFrame | null = null;
  let geoDirty = false;
  let newResult = false;
  let faceFound = false;
  let faceSince = 0;
  let pose: HeadPose | null = null;
  let gain = 1;
  let resultCapture = 0; // capture time (ms) of the frame behind the latest perception result
  let latestCapture = 0; // capture time (ms) of the newest camera frame (what the <video> shows)
  let resultHead: HeadProxy | null = null;
  let predictMs = 0;
  let shadowPendingSince: number | null = null;
  let shadowNeedsFull = false;
  const verts = new Float32Array(FACE_VERTEX_COUNT * 8);
  const predBuf = new Float32Array(FACE_VERTEX_COUNT * 3);

  const expBuf = new Float32Array(FACE_VERTEX_COUNT * 3);
  // Greenroom auto-fit of the face-mesh boundary: per-direction expansion measured from the seg
  // face-skin mask during capture (short hair → expand up to the hairline; hair over the forehead → stay tight).
  let fitProfile: Float32Array | null = null;
  let fitExpandAtFit = 0;
  let fitProfiles: Float32Array[] = [];
  let fitFrame = 0;
  let fitSummary = '';
  let maskReadBusy = false;
  let maskReadBuf: GPUBuffer | null = null;
  const collectFit = (mask: Uint8Array, mw: number, mh: number, lm: Float32Array) => {
    const prof = estimateFaceExpandProfile(mask, mw, mh, lm, procW, procH);
    if (prof) fitProfiles.push(prof);
  };
  /** Copy the GPU seg mask to the CPU (only during the 1.5 s Greenroom capture, every 3rd frame). */
  const readMaskForFit = (tex: GPUTexture, lm: Float32Array) => {
    if (maskReadBusy) return;
    const mw = tex.width, mh = tex.height, bpr = Math.ceil((mw * 4) / 256) * 256;
    if (!maskReadBuf || maskReadBuf.size !== bpr * mh) {
      maskReadBuf?.destroy();
      maskReadBuf = device.createBuffer({ size: bpr * mh, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
    const buf = maskReadBuf;
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [mw, mh]);
    device.queue.submit([enc.finish()]);
    maskReadBusy = true;
    const lmCopy = Float32Array.from(lm);
    buf.mapAsync(GPUMapMode.READ).then(() => {
      const src = new Uint8Array(buf.getMappedRange());
      const packed = bpr === mw * 4 ? new Uint8Array(src) : new Uint8Array(mw * mh * 4);
      if (bpr !== mw * 4) for (let y = 0; y < mh; y++) packed.set(src.subarray(y * bpr, y * bpr + mw * 4), y * mw * 4);
      buf.unmap();
      collectFit(packed, mw, mh, lmCopy);
    }).catch(() => {}).finally(() => (maskReadBusy = false));
  };
  const finishFit = () => {
    const med = medianProfile(fitProfiles);
    fitProfiles = [];
    if (!med) return;
    fitProfile = med;
    const sorted = Float32Array.from(med).sort();
    fitExpandAtFit = Math.round(sorted[Math.floor(sorted.length / 2)]! * 100) / 100;
    per.expand.set(Math.max(0, fitExpandAtFit));
    fitExpandAtFit = per.expand();
    const top = med[Math.floor(med.length * 0.75)]!;
    const side = med[Math.floor(med.length * 0.125)]!;
    fitSummary = `mesh fit  top ${top.toFixed(2)} · jaw ${side.toFixed(2)} · median ${fitExpandAtFit.toFixed(2)}`;
  };
  const applyLandmarks = (lm: Float32Array) => {
    // The landmark mesh outline sits inside the real face contour → expand its outer ring.
    let prof: Float32Array | null = null;
    if (per.autoFit() && fitProfile) {
      // The slider still works as a global offset on top of the fitted per-direction profile.
      const off = per.expand() - fitExpandAtFit;
      prof = off === 0 ? fitProfile : fitProfile.map((v) => v + off);
    }
    const meshLm = expandFaceSilhouette(lm, procW, procH, per.expand(), expBuf, prof);
    geo.setFaceVertices(buildFaceVertices(meshLm, procW, procH, verts));
    pose = headPoseFromLandmarks(lm, procW, procH);
    const head = headProxyFromLandmarks(lm, procW / procH, pose.yawDeg);
    geo.setHead(head);
    return head;
  };

  /** Common landmark ingestion for both face backends. */
  const ingestFace = (landmarks: Float32Array | null, captureMs: number) => {
    perfMeter.tick();
    euro.minCutoff = per.minCutoff();
    euro.beta = per.beta();
    euro.rigidSnap = per.snap() && sameFrame();
    resultCapture = captureMs;
    if (landmarks) {
      const sm = euro.filter(landmarks.subarray(0, FACE_VERTEX_COUNT * 3), captureMs / 1000);
      resultHead = applyLandmarks(sm);
      if (!faceFound) faceSince = performance.now();
      faceFound = true;
    } else {
      geo.setFaceVertices(null);
      geo.setHead(null);
      euro.reset();
      resultHead = null;
      faceFound = false;
      pose = null;
    }
    geo.opts.blobStrength = per.blob();
    geo.opts.faceFeatherPx = per.feather();
    geo.opts.headFlatten = per.flatten();
    geo.opts.relightHair = relightHair();
    // Inward feather avoids glow on the *camera* background; with a virtual background that background
    // is replaced anyway, and the feather would leave an un-relit (camera-lit) rim = a bright gap.
    geo.opts.edgeSoftPx = bgState.image ? Math.min(edgeSoft(), 2) : edgeSoft();
    geo.opts.normalSmoothPx = nSmooth();
    geo.maskShift = [0, 0];
    geoDirty = true;
    newResult = true;
  };

  let lastResultId = 0;
  const onResult = (r: PerceptionResult) => {
    if (r.id < lastResultId) return; // out-of-order (pipelined) result: never step back in time
    lastResultId = r.id;
    const t0 = performance.now();
    avg.face.push(r.faceMs);
    avg.seg.push(r.segMs);
    avg.lat.push(client?.latencyMs ?? 0);
    const cap = captureTimes.get(r.id) ?? performance.now();
    for (const id of captureTimes.keys()) if (id <= r.id) captureTimes.delete(id);
    if (useLitertFace()) return; // LiteRT face backend owns landmarks; ignore stray worker results
    ingestFace(r.landmarks, cap);
    if (r.mask && !useLitert()) geo.setMask(r.mask, r.maskWidth, r.maskHeight);
    if (greenroom.state === 'capturing' && per.autoFit() && r.mask && r.landmarks && !useLitert() && fitFrame++ % 3 === 0) collectFit(r.mask, r.maskWidth, r.maskHeight, r.landmarks);

    const f = frames.get(r.id);
    if (f) {
      frames.delete(r.id);
      display?.close();
      display = f;
    }
    for (const [id, old] of frames) if (id < r.id) { old.close(); frames.delete(id); }
    avg.cpuGeo.push(performance.now() - t0);
  };
  /** Start the MediaPipe worker on first need (fallback / user-selected backend). */
  const ensureClient = () => {
    if (client || clientLoading) return;
    clientLoading = true;
    PerceptionClient.create(ASSET_BASE).then((c) => {
      c.onError = (m) => (errEl.textContent = m);
      c.onResult = onResult;
      client = c;
    }, (e) => (errEl.textContent = `MediaPipe fallback failed: ${e}`));
  };
  if (!useLitertFace()) ensureClient();

  // Same-frame (Meet-style) mode: hold the camera frame, run seg + face on it, and render that exact
  // frame only once both are done. Adds the inference time (~12 ms) to the video but the light can
  // never trail the face. Requires the LiteRT backends (whole chain on our GPUDevice).
  const sameFrame = () => ui.latency() === 'frame' && useLitertFace();
  let sfBusy = false;
  let sfLastRender = 0;
  const sfWait = new RollingAvg(0.1);
  const runSameFrame = (cap: number) => {
    const s = seg!;
    const f = face!;
    sfBusy = true;
    const t0 = performance.now();
    const vf = new VideoFrame(video);
    const ext = device.importExternalTexture({ source: vf });
    const segDone = s.run(ext).then(() => {
      segMeter.tick();
      segGpu.push(s.timings.gpuMs);
    });
    const faceDone = f.run(ext, s.frameTensor);
    Promise.all([segDone, faceDone])
      .then(([, res]) => {
        if (res) {
          faceLm.push(f.timings.lmMs);
          ingestFace(res.landmarks, cap);
          if (greenroom.state === 'capturing' && per.autoFit() && fitFrame++ % 3 === 0) readMaskForFit(s.mask, res.landmarks);
        } else if (!f.roi && faceFound) {
          ingestFace(null, cap);
        }
        geoDirty = true;
        display?.close();
        display = vf;
        sfWait.push(performance.now() - t0);
        renderFrame();
        sfLastRender = performance.now();
      }, (e) => {
        vf.close();
        errEl.textContent = String(e?.stack ?? e);
      })
      .finally(() => (sfBusy = false));
  };

  const onVideoFrame = (_now: number, meta: VideoFrameCallbackMetadata) => {
    const cap = meta.captureTime ?? performance.now();
    camMeter.tick();
    if (meta.captureTime !== undefined) capToShow.push(meta.expectedDisplayTime - meta.captureTime);
    latestCapture = cap;
    if (sameFrame()) {
      if (!sfBusy && !seg!.busy && !face!.busy) runSameFrame(cap);
    } else if (!useLitertFace()) {
      ensureClient();
      const c = client;
      if (c && !c.busy) {
        const id = nextId++;
        const vf = new VideoFrame(video);
        frames.set(id, vf);
        captureTimes.set(id, cap);
        c.submit(vf, id, procW, procH).catch((e) => (errEl.textContent = String(e)));
      }
    }
    video.requestVideoFrameCallback(onVideoFrame);
  };
  video.requestVideoFrameCallback(onVideoFrame);

  const renderFrame = () => {
    meter.tick();
    const now = performance.now();
    const predictMode = ui.latency() === 'predict' || (ui.latency() === 'frame' && !sameFrame());
    // Low-latency mode: extrapolate landmarks to the capture time of the frame on screen
    // and rebuild geometry every render frame (60 Hz even when perception runs slower).
    if (predictMode && faceFound && euro.ready) {
      predictMs = Math.max(0, latestCapture - resultCapture) * per.predictGain();
      const lm = euro.predict(predictMs / 1000, predBuf);
      if (lm) {
        const head = applyLandmarks(lm);
        // The LiteRT mask is computed on the displayed frame → already aligned; only the MediaPipe mask lags.
        geo.maskShift = resultHead && !useLitert() ? [head.cx - resultHead.cx, head.cy - resultHead.cy] : [0, 0];
        geoDirty = true;
      }
    } else predictMs = 0;
    if (green.auto() && greenroom.state === 'idle' && faceFound && now - faceSince > 1000) greenroom.start();
    if (!green.heartbeat() && greenroom.state === 'ready') greenroom.opts.heartbeatMs = Infinity;
    else greenroom.opts.heartbeatMs = 5000;

    const profile = rig.buildProfile();
    if (dirty.has('bake') || greenroom.fading) {
      const t0 = performance.now();
      const sh = greenroom.roomSH(now);
      let room: RoomLight = { kind: 'flat', level: green.roomLevel() };
      if (green.delight() && sh && greenroom.samples.length) {
        const norm = exposureNormalizedRoom(sh, greenroom.samples, profile.lighting_rig, green.exposure());
        room = { kind: 'sh', sh: norm.sh };
        gain = norm.gain;
      }
      relight.bake(profile, room);
      drawMatcapPreview(matcapCanvas, relight.lastBake!);
      avg.bake.push(performance.now() - t0);
      dirty.add('params');
    }
    if (dirty.has('shadow')) {
      shadowPendingSince = now;
      shadowNeedsFull = true;
    }
    if (shadowPendingSince !== null) {
      const key = profile.lighting_rig.lights.find((l) => l.role === 'key')!;
      const dir = lightDirection(key.azimuth_deg, key.elevation_deg);
      if (now - shadowPendingSince > 150 && shadowNeedsFull) {
        shadows.bake(dir, key.softness, shadowSamples()); // settled: full quality
        shadowNeedsFull = false;
        shadowPendingSince = null;
      } else if (dirty.has('shadow')) {
        shadows.bake(dir, key.softness, 2); // while dragging: cheap preview
      }
    }
    if (dirty.has('bake') || dirty.has('params')) {
      dome.visible = ui.dome();
      if (ui.dome()) dome.update(profile, ui.mirror());
    }
    // Shadow layer weights follow head yaw every frame.
    if (bgState.dirty) {
      relight.setBackground(bgState.image);
      bgState.dirty = false;
    }
    relight.setParams({
      ...DEFAULT_PARAMS,
      lightWrap: bgWrap(),
      mode: Number(ui.mode()) as ViewMode,
      splitX: ui.split(),
      strength: ui.strength(),
      mirror: ui.mirror(),
      roughness: rig.roughness(),
      specStrength: rig.spec(),
      rimColor: rig.rimColor(),
      shadowStrength: shadowStrength(),
      edgeFade: bgState.image ? edgeFade() * 0.3 : edgeFade(),
      highlightSoft: hlSoft(),
      refinedMatte: !!bgState.image && bgRefine(),
      shadowWeights: shadowOn() && pose ? shadowLayerWeights(pose.yawDeg) : [0, 0, 0],
    });
    dirty.clear();

    const encoder = device.createCommandEncoder();
    const useDisplay = sameFrame() || (ui.latency() === 'sync' && !useLitertFace());
    const source = useDisplay && display ? display : video;
    const ext = device.importExternalTexture({ source });
    if (client) client.segment = !useLitert();
    if (useLitert() && seg) geo.useExternalMask(seg.mask);
    let afterGather: (() => void) | null = null;
    if (geoDirty) {
      geo.encode(encoder);
      geoDirty = false;
      if (newResult) {
        newResult = false;
        afterGather = greenroom.tick(encoder, ext, geo.geo, geo.mask.createView(), faceFound);
      }
    }
    const useRefine = !!bgState.image && bgRefine();
    if (useRefine) {
      matte.opts.temporal = bgTemporal() ? 0.45 : 1;
      matte.encode(encoder, ext, geo.mask);
    }
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      ...timer.timestampWrites(),
    });
    relight.draw(pass, {
      source: ext,
      geo: geo.geo.createView(),
      mask: geo.mask.createView(),
      faceUv: geo.faceUv.createView(),
      shadow: shadows.view,
      matte: useRefine ? matte.matte.createView() : undefined,
    });
    pass.end();
    const after = timer.resolve(encoder);
    // Present first: WebGPU has one queue, so anything submitted before this (model inference)
    // would delay the visible frame even though fps stays at 60.
    device.queue.submit([encoder.finish()]);
    after();
    afterGather?.();

    // Then kick off perception on the same camera frame; results are used from the next frame on.
    if (!sameFrame() && useLitert() && seg && !seg.busy) {
      const s = seg;
      void s.run(ext).then(() => {
        segMeter.tick();
        segGpu.push(s.timings.gpuMs);
        geoDirty = true;
      });
    }
    if (!sameFrame() && useLitertFace() && face && seg && !face.busy) {
      const cap = latestCapture;
      const f = face;
      void f.run(ext, seg.frameTensor).then((res) => {
        if (res) {
          faceLm.push(f.timings.lmMs);
          ingestFace(res.landmarks, cap);
        } else if (!f.roi && faceFound) {
          ingestFace(null, cap);
        }
      }, (e) => (errEl.textContent = String(e?.stack ?? e)));
    }

    if (greenroom.state === 'capturing') {
      banner.style.display = 'block';
      banner.textContent = `Greenroom ${(greenroom.progress * 100).toFixed(0)}% — slowly turn your head left ↔ right`;
    } else banner.style.display = 'none';
    const fit = greenroom.fit;
    grStats.textContent = [
      `state     ${greenroom.state}${greenroom.fading ? ' (cross-fading)' : ''}`,
      fit ? `samples   ${fit.inlierCount}/${fit.sampleCount} inliers` : '',
      fit ? `residual  ${(fit.rmsRel * 100).toFixed(1)}% rms   coverage ${(fit.coverage * 100).toFixed(0)}%` : '',
      fit ? `gain ${gain.toFixed(2)}   refits ${greenroom.refits}` : '',
      greenroom.lastEvent,
      per.autoFit() ? fitSummary || 'mesh fit  (pending)' : '',
    ].filter(Boolean).join('\n');

    hud.textContent = hudExpanded ? [
      `M3 Greenroom relight   ${W}x${H}  (perception ${procW}x${procH}, geo ${geo.width}x${geo.height})`,
      `render fps        ${meter.fps.toFixed(1)}`,
      `camera fps        ${camMeter.fps.toFixed(1)}   camera→screen ${capToShow.value ? capToShow.value.toFixed(0) + ' ms' : 'n/a'}`,
      `perception fps    ${perfMeter.fps.toFixed(1)}`,
      `relight pass GPU  ${timer.enabled ? timer.avgMs.toFixed(3) + ' ms' : 'n/a'}`,
      `segmentation      ${useLitert() ? `LiteRT.js WebGPU ${segMeter.fps.toFixed(0)} fps, chain ${segGpu.value.toFixed(1)} ms, no readback` : "MediaPipe (worker)"}`,
      `face              ${useLitertFace() ? `LiteRT.js WebGPU landmarks ${faceLm.value.toFixed(1)} ms (det ${face!.detections}x, ${face!.timings.detMs.toFixed(1)} ms)` : "MediaPipe (worker)"}`,
      `worker face/seg   ${avg.face.value.toFixed(1)} / ${avg.seg.value.toFixed(1)} ms   round-trip ${avg.lat.value.toFixed(1)} ms`,
      `latency mode      ${ui.latency()}${predictMode ? `  (predicting +${predictMs.toFixed(0)} ms)` : ''}${sameFrame() ? `  (frame held ${sfWait.value.toFixed(1)} ms for inference)` : ''}`,
      `matcap bake CPU   ${avg.bake.value.toFixed(2)} ms`,
      `face              ${faceFound ? 'yes' : 'no'}${pose ? `  yaw ${pose.yawDeg.toFixed(0)}°  pitch ${pose.pitchDeg.toFixed(0)}°` : ''}`,
      `room              ${green.delight() && greenroom.fit ? 'fitted SH (de-light + add-light)' : 'flat (add-light only)'}`,
    ].join('\n') :
      `${camMeter.fps.toFixed(0)} fps · ${sameFrame() ? `+${sfWait.value.toFixed(0)} ms` : ui.latency()} · face ${faceFound ? '✓' : '—'} · ${greenroom.fit ? 'room fitted' : greenroom.state}  ▸`;
  };
  const loop = () => {
    // Same-frame mode renders from the inference completion; rAF only covers camera stalls.
    if (!sameFrame() || performance.now() - sfLastRender > 200) renderFrame();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

main().catch((e: unknown) => {
  console.error(e);
  errEl.textContent = String(e instanceof Error ? e.stack ?? e.message : e);
});
