import type { LightingPreset } from './ui.ts';

const toLin = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const toSrgb = (c: number) => (c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055);
const hex = (rgb: number[]) =>
  '#' + rgb.map((c) => Math.round(Math.min(1, Math.max(0, toSrgb(c))) * 255).toString(16).padStart(2, '0')).join('');

/** Normalise a linear colour to max channel 1 and pull it `keep` of the way from white (avoid extreme gels). */
function gel(rgb: number[], keep: number): number[] {
  const m = Math.max(...rgb, 1e-4);
  return rgb.map((c) => 1 - keep + keep * (c / m));
}

export interface BackgroundLightEstimate {
  preset: LightingPreset;
  /** Debug: where the key was found (image uv) and how contrasty the scene is. */
  keyUv: [number, number];
  contrast: number;
}

/**
 * Which side of a portrait's subject is lit: mean luminance of the left vs right half of the central
 * region (where the subject usually is). Returns null when the halves are within 20 % of each other.
 */
export function measureLitSide(img: ImageBitmap): { side: 'left' | 'right' | null; ratio: number } {
  const W = 64, H = 64;
  const cv = new OffscreenCanvas(W, H);
  const ctx = cv.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0, W, H);
  const px = ctx.getImageData(0, 0, W, H).data;
  let l = 0, r = 0;
  for (let y = Math.floor(H * 0.15); y < H * 0.85; y++) {
    for (let x = Math.floor(W * 0.2); x < W * 0.8; x++) {
      const i = (y * W + x) * 4;
      const v = 0.2126 * toLin(px[i]! / 255) + 0.7152 * toLin(px[i + 1]! / 255) + 0.0722 * toLin(px[i + 2]! / 255);
      if (x < W / 2) l += v; else r += v;
    }
  }
  const ratio = Math.max(l, r) / Math.max(Math.min(l, r), 1e-6);
  return { side: ratio < 1.2 ? null : l > r ? 'left' : 'right', ratio };
}

/**
 * Estimate a 3-light rig that matches a background image (a 2D picture of what is *behind* you):
 *  - key   : direction from the luminance-weighted centroid of the brightest 4 % of pixels (screen side →
 *            azimuth, height → elevation), colour = their mean colour, softness from scene contrast
 *            (a sun/lamp = hard, an overcast sky / office = soft);
 *  - rim   : a backlight on the same side as the source (bright sources in the background sit behind you);
 *  - fill  : from the other side, coloured by the darker half of the image, strength from its luminance;
 *  - ambient from the mean luminance.
 * `mirrored` = the view is a selfie mirror (rig azimuths are flipped relative to the screen).
 */
export function estimateLightingFromBackground(img: ImageBitmap, mirrored: boolean): BackgroundLightEstimate {
  const W = 64, H = 36;
  const cv = new OffscreenCanvas(W, H);
  const ctx = cv.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0, W, H);
  const px = ctx.getImageData(0, 0, W, H).data;
  const n = W * H;
  const lin = new Float32Array(n * 3);
  const lum = new Float32Array(n);
  let meanL = 0;
  const mean = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const v = toLin(px[i * 4 + c]! / 255);
      lin[i * 3 + c] = v;
      mean[c]! += v;
    }
    lum[i] = 0.2126 * lin[i * 3]! + 0.7152 * lin[i * 3 + 1]! + 0.0722 * lin[i * 3 + 2]!;
    meanL += lum[i]!;
  }
  meanL /= n;
  for (let c = 0; c < 3; c++) mean[c] = mean[c]! / n;
  const sorted = Float32Array.from(lum).sort();
  const hi = sorted[Math.floor(n * 0.96)]!;
  const median = sorted[Math.floor(n * 0.5)]!;

  // Key: brightest pixels.
  let wx = 0, wy = 0, ws = 0;
  const keyC = [0, 0, 0];
  const fillC = [0, 0, 0];
  let fillN = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const l = lum[i]!;
      if (l >= hi) {
        const w = l;
        wx += (x + 0.5) / W * w;
        wy += (y + 0.5) / H * w;
        ws += w;
        for (let c = 0; c < 3; c++) keyC[c]! += lin[i * 3 + c]! * w;
      } else if (l <= median) {
        for (let c = 0; c < 3; c++) fillC[c]! += lin[i * 3 + c]!;
        fillN++;
      }
    }
  }
  const cx = ws > 0 ? wx / ws : 0.5;
  const cy = ws > 0 ? wy / ws : 0.3;
  for (let c = 0; c < 3; c++) keyC[c] = keyC[c]! / Math.max(ws, 1e-6);
  for (let c = 0; c < 3; c++) fillC[c] = fillC[c]! / Math.max(fillN, 1);

  // Light from the *other* side of the frame: its brightest 10 % (e.g. the cyan side of a neon street,
  // a window opposite a lamp). Falls back to the darker half's colour when that side is dull.
  const otherLeft = cx >= 0.5;
  const otherLum: number[] = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if ((x < W / 2) === otherLeft) otherLum.push(lum[y * W + x]!);
  otherLum.sort((a, b) => a - b);
  const otherHi = otherLum[Math.floor(otherLum.length * 0.9)] ?? 0;
  const otherC = [0, 0, 0];
  let otherW = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if ((x < W / 2) !== otherLeft) continue;
      const i = y * W + x;
      if (lum[i]! >= otherHi) { for (let c = 0; c < 3; c++) otherC[c]! += lin[i * 3 + c]! * lum[i]!; otherW += lum[i]!; }
    }
  }
  for (let c = 0; c < 3; c++) otherC[c] = otherC[c]! / Math.max(otherW, 1e-6);
  const otherRatio = otherHi / Math.max(hi, 1e-4); // 0 … 1
  const sideFill = otherRatio > 0.35 ? otherC : fillC;
  // Saturated scenes (neon, sunset) keep more of their colour in the gels.
  const chroma = (c: number[]) => (Math.max(...c) - Math.min(...c)) / Math.max(Math.max(...c), 1e-4);
  const keep = 0.45 + 0.4 * Math.min(1, Math.max(chroma(keyC), chroma(sideFill)));

  const contrast = hi / Math.max(meanL, 1e-4); // ~1.5 flat … >5 sun/lamp
  const hard = Math.min(1, Math.max(0, (contrast - 1.6) / 4));
  // Screen side → rig azimuth (selfie mirror flips it). Keep the key in front of the face (±25…75°).
  const side = cx - 0.5; // −0.5 left … +0.5 right on screen
  const azMag = 25 + Math.min(1, Math.abs(side) * 2.2) * 50;
  const screenRightSign = side >= 0 ? 1 : -1;
  // Mirrored view: screen-right ↔ negative rig azimuth.
  const keyAz = Math.round((mirrored ? -screenRightSign : screenRightSign) * azMag);
  const keyEl = Math.round(Math.min(65, Math.max(-5, (0.55 - cy) * 110)));
  const brightness = Math.min(1, meanL * 3 + 0.25); // dark scenes → dimmer rig overall

  const preset: LightingPreset = {
    name: '🪄 Auto (from background)',
    key: {
      az: keyAz, el: keyEl,
      intensity: +((0.75 + 0.55 * hard) * brightness + 0.25).toFixed(2),
      softness: +(0.95 - 0.5 * hard).toFixed(2),
      kelvin: 6500,
      gel: hex(gel(keyC, keep)),
    },
    fill: {
      az: -Math.sign(keyAz || 1) * 55, el: 5,
      // A strong opposite-side source (two-colour neon) → strong fill; otherwise ambient-ish.
      intensity: +Math.min(0.9, 0.12 + Math.max(median / Math.max(hi, 1e-4), otherRatio * 0.8) * 0.9).toFixed(2),
      softness: 1,
      kelvin: 6500,
      gel: hex(gel(sideFill, keep)),
    },
    rim: {
      az: Math.sign(keyAz || 1) * 150, el: Math.max(10, keyEl),
      intensity: +(0.25 + 0.9 * hard * brightness).toFixed(2),
      kelvin: 6500,
      gel: hex(gel(keyC, Math.min(0.9, keep + 0.15))),
    },
    ambient: +Math.min(0.25, 0.04 + meanL * 0.5).toFixed(3),
    roughness: 0.5,
    spec: +(0.1 + 0.15 * hard).toFixed(2),
  };
  return { preset, keyUv: [cx, cy], contrast };
}
