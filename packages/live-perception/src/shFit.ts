/**
 * Greenroom room-light estimation (design doc §2.3 Stage A).
 *
 * Model per skin sample p:  I_c(p) ≈ ρ_c · Σ_i c'_i,c · Y_i(N(p))
 * Skin albedo is assumed ~constant over the face, so we fit S'(N) = ρ ⊙ S_room(N)
 * from many (normal, colour) samples gathered while the user turns their head
 * (multi-view → wider normal coverage, i.e. photometric-stereo-like constraints).
 *
 * Robustness: weighted ridge least-squares (band-1/2 regularised towards an
 * optional prior), then a second pass rejecting residual outliers (MAD).
 */
import { shBasis, type SH9 } from '@relight/relight-shader';

/** Packed samples: [nx, ny, nz, r, g, b] linear RGB, per sample. */
export const SAMPLE_STRIDE = 6;

export interface FitOptions {
  /** Ridge strength for bands 1 and 2 (relative to mean diagonal). */
  ridge?: [number, number];
  /** Prior coefficients to regularise towards (heartbeat refits). */
  prior?: SH9;
  /** Outlier rejection threshold in robust sigmas. */
  rejectSigma?: number;
}

export interface RoomFit {
  /** S' = ρ ⊙ S_room, raw fit in observed units. */
  shRaw: SH9;
  /** Room shading with mean skin chroma divided out (keeps skin colour, removes directional colour casts). */
  shRoom: SH9;
  sampleCount: number;
  inlierCount: number;
  /** RMS relative luminance residual of inliers. */
  rmsRel: number;
  /** Fraction of the camera-facing hemisphere covered by sample normals (on a 12×12 grid). */
  coverage: number;
  meanLum: number;
}

const lum = (r: number, g: number, b: number) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

/** Solve symmetric positive-definite 9x9 system M x = B (B has 3 columns) via Cholesky. */
function solveSpd(M: Float64Array, B: Float64Array): Float64Array {
  const n = 9;
  const L = new Float64Array(81);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = M[i * n + j]!;
      for (let k = 0; k < j; k++) s -= L[i * n + k]! * L[j * n + k]!;
      L[i * n + j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j * n + j]!;
    }
  }
  const X = new Float64Array(27);
  for (let c = 0; c < 3; c++) {
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = B[i * 3 + c]!;
      for (let k = 0; k < i; k++) s -= L[i * n + k]! * y[k]!;
      y[i] = s / L[i * n + i]!;
    }
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i]!;
      for (let k = i + 1; k < n; k++) s -= L[k * n + i]! * X[k * 3 + c]!;
      X[i * 3 + c] = s / L[i * n + i]!;
    }
  }
  return X;
}

const BAND = [0, 1, 1, 1, 2, 2, 2, 2, 2];

function solve(samples: Float32Array, keep: Uint8Array | null, opts: FitOptions): SH9 {
  const M = new Float64Array(81);
  const B = new Float64Array(27);
  const y = new Float32Array(9);
  const count = samples.length / SAMPLE_STRIDE;
  let used = 0;
  for (let s = 0; s < count; s++) {
    if (keep && !keep[s]) continue;
    const o = s * SAMPLE_STRIDE;
    shBasis([samples[o]!, samples[o + 1]!, samples[o + 2]!], y);
    for (let i = 0; i < 9; i++) {
      const yi = y[i]!;
      for (let j = 0; j <= i; j++) M[i * 9 + j] = M[i * 9 + j]! + yi * y[j]!;
      for (let c = 0; c < 3; c++) B[i * 3 + c] = B[i * 3 + c]! + yi * samples[o + 3 + c]!;
    }
    used++;
  }
  for (let i = 0; i < 9; i++) for (let j = 0; j < i; j++) M[j * 9 + i] = M[i * 9 + j]!;
  let diag = 0;
  for (let i = 0; i < 9; i++) diag += M[i * 9 + i]!;
  diag = diag / 9 || 1;
  const [r1, r2] = opts.ridge ?? [0.001, 0.005];
  for (let i = 0; i < 9; i++) {
    const lam = BAND[i] === 0 ? 1e-6 * diag : (BAND[i] === 1 ? r1 : r2) * diag;
    M[i * 9 + i] = M[i * 9 + i]! + lam;
    if (opts.prior) for (let c = 0; c < 3; c++) B[i * 3 + c] = B[i * 3 + c]! + lam * opts.prior[i * 3 + c]!;
  }
  if (used < 20) throw new Error(`Greenroom fit: too few skin samples (${used})`);
  return Float32Array.from(solveSpd(M, B));
}

function evalSh(sh: SH9, y: Float32Array, c: number): number {
  let v = 0;
  for (let i = 0; i < 9; i++) v += sh[i * 3 + c]! * y[i]!;
  return v;
}

export function fitRoomLight(samples: Float32Array, opts: FitOptions = {}): RoomFit {
  const count = samples.length / SAMPLE_STRIDE;
  let sh = solve(samples, null, opts);

  // Pass 2: reject outliers (eyes, lips, beard, specular highlights) by relative luminance residual.
  const y = new Float32Array(9);
  const res = new Float32Array(count);
  for (let s = 0; s < count; s++) {
    const o = s * SAMPLE_STRIDE;
    shBasis([samples[o]!, samples[o + 1]!, samples[o + 2]!], y);
    const pred = lum(evalSh(sh, y, 0), evalSh(sh, y, 1), evalSh(sh, y, 2));
    const obs = lum(samples[o + 3]!, samples[o + 4]!, samples[o + 5]!);
    res[s] = (obs - pred) / Math.max(pred, 1e-3);
  }
  const sorted = Float32Array.from(res).map(Math.abs).sort();
  const mad = sorted[Math.floor(sorted.length / 2)]! || 1e-3;
  const thr = (opts.rejectSigma ?? 2.5) * 1.4826 * mad;
  const keep = new Uint8Array(count);
  let inliers = 0;
  let sq = 0;
  for (let s = 0; s < count; s++) {
    if (Math.abs(res[s]!) <= thr) {
      keep[s] = 1;
      inliers++;
      sq += res[s]! * res[s]!;
    }
  }
  sh = solve(samples, keep, opts);

  // Coverage of camera-facing normals on a 12x12 grid over (nx, ny).
  const G = 12;
  const grid = new Uint8Array(G * G);
  let gridTotal = 0;
  for (let j = 0; j < G; j++)
    for (let i = 0; i < G; i++) {
      const x = ((i + 0.5) / G) * 2 - 1;
      const yy = ((j + 0.5) / G) * 2 - 1;
      if (x * x + yy * yy < 0.9) gridTotal++;
    }
  let meanLum = 0;
  for (let s = 0; s < count; s++) {
    const o = s * SAMPLE_STRIDE;
    const gi = Math.min(G - 1, Math.floor((samples[o]! * 0.5 + 0.5) * G));
    const gj = Math.min(G - 1, Math.floor((samples[o + 1]! * 0.5 + 0.5) * G));
    grid[gj * G + gi] = 1;
    meanLum += lum(samples[o + 3]!, samples[o + 4]!, samples[o + 5]!);
  }
  meanLum /= Math.max(count, 1);
  const coverage = Math.min(1, grid.reduce((a, b) => a + b, 0) / gridTotal);

  // Divide out mean skin chroma (DC colour) so skin keeps its observed colour.
  const d = [sh[0]!, sh[1]!, sh[2]!];
  const dl = lum(d[0]!, d[1]!, d[2]!) || 1;
  const k = d.map((v) => Math.max(v, 1e-4) / dl);
  const shRoom = new Float32Array(27);
  for (let i = 0; i < 9; i++) for (let c = 0; c < 3; c++) shRoom[i * 3 + c] = sh[i * 3 + c]! / k[c]!;

  return {
    shRaw: sh,
    shRoom,
    sampleCount: count,
    inlierCount: inliers,
    rmsRel: Math.sqrt(sq / Math.max(inliers, 1)),
    coverage,
    meanLum,
  };
}

/**
 * Exposure normalisation: returns gain g such that relighting with
 * Q = g · T(N) / S_room(N) preserves the mean face luminance of the samples.
 * `ratioLum(n)` must return lum(T(n)/S_room(n)) (without g).
 */
export function exposureGain(samples: Float32Array, ratioLum: (n: [number, number, number]) => number, maxSamples = 3000): number {
  const count = samples.length / SAMPLE_STRIDE;
  const step = Math.max(1, Math.floor(count / maxSamples));
  let a = 0;
  let b = 0;
  for (let s = 0; s < count; s += step) {
    const o = s * SAMPLE_STRIDE;
    const l = lum(samples[o + 3]!, samples[o + 4]!, samples[o + 5]!);
    a += l;
    b += l * ratioLum([samples[o]!, samples[o + 1]!, samples[o + 2]!]);
  }
  return b > 0 ? a / b : 1;
}
