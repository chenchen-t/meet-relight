/**
 * Pre-Integrated Skin Shading (Penner 2011) with the d'Eon/Luebke 6-Gaussian
 * skin diffusion profile. Produces a per-channel 1D LUT D(N·L) for a fixed
 * surface curvature radius r (mm):
 *
 *   D(θ) = ∫ max(cos(θ + x), 0) · R(2r·sin(x/2)) dx  /  ∫ R(2r·sin(x/2)) dx
 *
 * Small r (nose, lips) → strong red bleed past the terminator; large r → Lambert.
 * Replaces the per-channel wrap approximation used in M1.
 */
import type { Vec3 } from './vec.ts';

/** d'Eon & Luebke (GPU Gems 3, ch.14) skin profile: [variance mm², weight R, G, B]. */
const SKIN_PROFILE: [number, number, number, number][] = [
  [0.0064, 0.233, 0.455, 0.649],
  [0.0484, 0.1, 0.336, 0.344],
  [0.187, 0.118, 0.198, 0.0],
  [0.567, 0.113, 0.007, 0.007],
  [1.99, 0.358, 0.004, 0.0],
  [7.41, 0.078, 0.0, 0.0],
];

function profile(d: number, c: number): number {
  let s = 0;
  for (const g of SKIN_PROFILE) {
    const v = g[0];
    s += (g[c + 1]! / (2 * Math.PI * v)) * Math.exp((-d * d) / (2 * v));
  }
  return s;
}

export const SSS_LUT_SIZE = 128;

export interface SssLut {
  radiusMm: number;
  /** SSS_LUT_SIZE entries × 3 channels, indexed by (N·L * 0.5 + 0.5). */
  data: Float32Array;
}

const cache = new Map<number, SssLut>();

export function preintegratedSkinLut(radiusMm: number): SssLut {
  const key = Math.round(radiusMm * 100) / 100;
  const hit = cache.get(key);
  if (hit) return hit;
  const data = new Float32Array(SSS_LUT_SIZE * 3);
  const steps = 200; // one-off per radius (cached); max abs diff vs 720 steps = 0.0016
  const dx = (2 * Math.PI) / steps;
  for (let i = 0; i < SSS_LUT_SIZE; i++) {
    const ndl = (i / (SSS_LUT_SIZE - 1)) * 2 - 1;
    const theta = Math.acos(ndl);
    for (let c = 0; c < 3; c++) {
      let num = 0;
      let den = 0;
      for (let k = 0; k < steps; k++) {
        const x = -Math.PI + (k + 0.5) * dx;
        const w = profile(Math.abs(2 * key * Math.sin(x / 2)), c);
        num += Math.max(Math.cos(theta + x), 0) * w;
        den += w;
      }
      data[i * 3 + c] = num / den;
    }
  }
  const lut = { radiusMm: key, data };
  cache.set(key, lut);
  return lut;
}

export function sampleSssLut(lut: SssLut, ndl: number, out: Vec3 = [0, 0, 0]): Vec3 {
  const f = Math.min(1, Math.max(0, ndl * 0.5 + 0.5)) * (SSS_LUT_SIZE - 1);
  const i0 = Math.floor(f);
  const i1 = Math.min(SSS_LUT_SIZE - 1, i0 + 1);
  const a = f - i0;
  for (let c = 0; c < 3; c++) out[c] = lut.data[i0 * 3 + c]! * (1 - a) + lut.data[i1 * 3 + c]! * a;
  return out;
}
