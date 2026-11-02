/**
 * Order-2 real spherical harmonics (9 coefficients per colour channel).
 *
 * We store *shading* SH directly: shading(N) = Σ_i c_i · Y_i(N), i.e. the
 * cosine convolution is already folded in. This is the representation the
 * Greenroom fitter (M3) estimates from pixels, and what the quotient baker
 * divides by.
 */
import type { Vec3 } from './vec.ts';

export type SH9 = Float32Array; // length 27: [c0.rgb, c1.rgb, ... c8.rgb]

export function shBasis(n: Vec3, out: Float32Array = new Float32Array(9)): Float32Array {
  const [x, y, z] = n;
  out[0] = 0.282095;
  out[1] = 0.488603 * y;
  out[2] = 0.488603 * z;
  out[3] = 0.488603 * x;
  out[4] = 1.092548 * x * y;
  out[5] = 1.092548 * y * z;
  out[6] = 0.315392 * (3 * z * z - 1);
  out[7] = 1.092548 * x * z;
  out[8] = 0.546274 * (x * x - y * y);
  return out;
}

const tmpBasis = new Float32Array(9);

export function shEval(sh: SH9, n: Vec3, out: Vec3 = [0, 0, 0]): Vec3 {
  const b = shBasis(n, tmpBasis);
  let r = 0;
  let g = 0;
  let bl = 0;
  for (let i = 0; i < 9; i++) {
    r += sh[i * 3]! * b[i]!;
    g += sh[i * 3 + 1]! * b[i]!;
    bl += sh[i * 3 + 2]! * b[i]!;
  }
  out[0] = r;
  out[1] = g;
  out[2] = bl;
  return out;
}

/** Deterministic, near-uniform points on the unit sphere. */
export function fibonacciSphere(count: number): Vec3[] {
  const pts: Vec3[] = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i++) {
    const y = 1 - (2 * (i + 0.5)) / count;
    const r = Math.sqrt(1 - y * y);
    const t = golden * i;
    pts.push([Math.cos(t) * r, y, Math.sin(t) * r]);
  }
  return pts;
}

/** Project an arbitrary RGB shading function over the sphere into SH9 (Monte Carlo on a Fibonacci grid). */
export function shProject(fn: (n: Vec3) => Vec3, samples = 4096): SH9 {
  const sh = new Float32Array(27);
  const b = new Float32Array(9);
  const w = (4 * Math.PI) / samples;
  for (const n of fibonacciSphere(samples)) {
    const v = fn(n);
    shBasis(n, b);
    for (let i = 0; i < 9; i++) {
      sh[i * 3] = sh[i * 3]! + v[0] * b[i]! * w;
      sh[i * 3 + 1] = sh[i * 3 + 1]! + v[1] * b[i]! * w;
      sh[i * 3 + 2] = sh[i * 3 + 2]! + v[2] * b[i]! * w;
    }
  }
  return sh;
}

/** Pack SH9 into 9 vec4 (xyz used) for a WGSL uniform `array<vec4f, 9>`. */
export function shToVec4Array(sh: SH9): Float32Array {
  const out = new Float32Array(36);
  for (let i = 0; i < 9; i++) {
    out[i * 4] = sh[i * 3]!;
    out[i * 4 + 1] = sh[i * 3 + 1]!;
    out[i * 4 + 2] = sh[i * 3 + 2]!;
  }
  return out;
}
