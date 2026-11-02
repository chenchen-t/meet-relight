/**
 * CPU-side face geometry: MediaPipe landmarks → smoothed camera-space vertices,
 * per-vertex normals and head pose. ~0.1 ms for 468 verts / 898 tris.
 */
import { FACE_TRIANGLES, FACE_VERTEX_COUNT, CANONICAL_UVS } from './faceMeshData.ts';

/** Row-major 3×3 inverse, or null if (near-)singular. */
function invert3(m: Float64Array): Float64Array | null {
  const [a, b, c, d, e, f, g, h, i] = m as unknown as number[];
  const A = e! * i! - f! * h!, B = -(d! * i! - f! * g!), C = d! * h! - e! * g!;
  const det = a! * A + b! * B + c! * C;
  if (!(Math.abs(det) > 1e-18)) return null;
  const s = 1 / det;
  return new Float64Array([
    A * s, -(b! * i! - c! * h!) * s, (b! * f! - c! * e!) * s,
    B * s, (a! * i! - c! * g!) * s, -(a! * f! - c! * d!) * s,
    C * s, -(a! * h! - b! * g!) * s, (a! * e! - b! * d!) * s,
  ]);
}

/**
 * One-Euro filter over a flat Float32Array (Casiez et al. 2012): low jitter at rest,
 * low lag in motion. Also tracks the velocity of the *filtered* signal so callers can
 * extrapolate forward (`predict`) to hide perception + smoothing latency.
 */
export class OneEuroArray {
  private x: Float32Array | null = null;
  private dx: Float32Array | null = null;
  private vel: Float32Array | null = null;
  private out: Float32Array | null = null;
  private tPrev = 0;
  /**
   * Motion-compensated filtering for xyz point sets (stride 3): before filtering, the state is warped
   * by the least-squares 2D affine motion to the raw frame, so translation / scale / roll and most of
   * a head turn's foreshortening pass through with zero lag (no "light trails the face when I start
   * moving"), while per-vertex jitter is still smoothed. Turn off when using `predict`.
   */
  public rigidSnap = true;
  constructor(
    public minCutoff = 1.5,
    public beta = 20,
    public dCutoff = 4.0,
    /** EMA factor for the prediction velocity (higher = more responsive, noisier). */
    public velAlpha = 0.6,
  ) {}

  private static alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  reset(): void {
    this.x = null;
  }

  get ready(): boolean {
    return !!this.x;
  }

  filter(v: Float32Array, tSec: number): Float32Array {
    if (!this.x || this.x.length !== v.length) {
      this.x = Float32Array.from(v);
      this.dx = new Float32Array(v.length);
      this.vel = new Float32Array(v.length);
      this.out = new Float32Array(v.length);
      this.tPrev = tSec;
      return this.x;
    }
    const dt = Math.max(1e-3, tSec - this.tPrev);
    this.tPrev = tSec;
    const ad = OneEuroArray.alpha(this.dCutoff, dt);
    const x = this.x;
    const dx = this.dx!;
    const vel = this.vel!;
    const va = this.velAlpha;
    // Motion compensation: carry the filter state along with this frame's affine motion so the
    // One-Euro only sees the non-affine residual (jitter, expression). Keep the pre-warp copy for `vel`.
    const prev = this.out!;
    prev.set(x);
    if (this.rigidSnap && v.length % 3 === 0) this.warpAffine(x, v);
    for (let i = 0; i < v.length; i++) {
      const d = (v[i]! - x[i]!) / dt;
      dx[i] = dx[i]! + ad * (d - dx[i]!);
      const a = OneEuroArray.alpha(this.minCutoff + this.beta * Math.abs(dx[i]!), dt);
      const nx = x[i]! + a * (v[i]! - x[i]!);
      vel[i] = vel[i]! + va * ((nx - prev[i]!) / dt - vel[i]!);
      x[i] = nx;
    }
    return x;
  }

  /**
   * In place: x ← A·(x − mx) + mv with A the least-squares 3×3 linear map x→v. Using depth as an input
   * lets the map represent head rotation (yaw/pitch mix z into x/y), not just image-plane motion.
   */
  private warpAffine(x: Float32Array, v: Float32Array): void {
    const n = v.length / 3;
    const mf = [0, 0, 0], mr = [0, 0, 0];
    for (let i = 0; i < v.length; i += 3) {
      for (let k = 0; k < 3; k++) { mf[k]! += x[i + k]!; mr[k]! += v[i + k]!; }
    }
    for (let k = 0; k < 3; k++) { mf[k] = mf[k]! / n; mr[k] = mr[k]! / n; }
    const S = new Float64Array(9); // Σ f fᵀ
    const U = new Float64Array(9); // Σ r fᵀ
    for (let i = 0; i < v.length; i += 3) {
      const f0 = x[i]! - mf[0]!, f1 = x[i + 1]! - mf[1]!, f2 = x[i + 2]! - mf[2]!;
      const r0 = v[i]! - mr[0]!, r1 = v[i + 1]! - mr[1]!, r2 = v[i + 2]! - mr[2]!;
      const f = [f0, f1, f2], r = [r0, r1, r2];
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
        S[a * 3 + b]! += f[a]! * f[b]!;
        U[a * 3 + b]! += r[a]! * f[b]!;
      }
    }
    // Regularise towards identity (depth is noisier / flatter than xy), then A = U·S⁻¹.
    const tr = (S[0]! + S[4]! + S[8]!) * 1e-3;
    for (let k = 0; k < 3; k++) { S[k * 4]! += tr; U[k * 4]! += tr; }
    const inv = invert3(S);
    if (!inv) {
      for (let i = 0; i < v.length; i += 3) for (let k = 0; k < 3; k++) x[i + k] = x[i + k]! + mr[k]! - mf[k]!;
      return;
    }
    const A = new Float64Array(9);
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
      A[a * 3 + b] = U[a * 3]! * inv[b]! + U[a * 3 + 1]! * inv[3 + b]! + U[a * 3 + 2]! * inv[6 + b]!;
    }
    for (let i = 0; i < v.length; i += 3) {
      const f0 = x[i]! - mf[0]!, f1 = x[i + 1]! - mf[1]!, f2 = x[i + 2]! - mf[2]!;
      x[i] = A[0]! * f0 + A[1]! * f1 + A[2]! * f2 + mr[0]!;
      x[i + 1] = A[3]! * f0 + A[4]! * f1 + A[5]! * f2 + mr[1]!;
      x[i + 2] = A[6]! * f0 + A[7]! * f1 + A[8]! * f2 + mr[2]!;
    }
  }

  /** Filtered value extrapolated `dtSec` seconds ahead (clamped to 100 ms). */
  predict(dtSec: number, out?: Float32Array): Float32Array | null {
    if (!this.x || !this.vel) return null;
    const dt = Math.min(0.1, Math.max(0, dtSec));
    const o = out && out.length === this.x.length ? out : new Float32Array(this.x.length);
    for (let i = 0; i < this.x.length; i++) o[i] = this.x[i]! + this.vel[i]! * dt;
    return o;
  }
}

/**
 * Per-vertex buffer layout uploaded to the GPU face rasterizer (8 floats / vertex):
 *   [ndc.x, ndc.y, depth, n.x, n.y, n.z, uv.u, uv.v]
 * n is the camera-space unit normal (+x right, +y up, +z towards camera).
 */
export const FACE_VERTEX_STRIDE = 8;

/** Area-weighted unit vertex normals for the canonical topology. P: xyz interleaved (468 verts). */
export function computeVertexNormals(P: Float32Array, out: Float32Array = new Float32Array(FACE_VERTEX_COUNT * 3)): Float32Array {
  out.fill(0);
  const T = FACE_TRIANGLES;
  for (let t = 0; t < T.length; t += 3) {
    const a = T[t]! * 3;
    const b = T[t + 1]! * 3;
    const c = T[t + 2]! * 3;
    const e1x = P[b]! - P[a]!, e1y = P[b + 1]! - P[a + 1]!, e1z = P[b + 2]! - P[a + 2]!;
    const e2x = P[c]! - P[a]!, e2y = P[c + 1]! - P[a + 1]!, e2z = P[c + 2]! - P[a + 2]!;
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    for (const k of [a, b, c]) {
      out[k] = out[k]! + nx;
      out[k + 1] = out[k + 1]! + ny;
      out[k + 2] = out[k + 2]! + nz;
    }
  }
  for (let i = 0; i < FACE_VERTEX_COUNT; i++) {
    const l = Math.hypot(out[i * 3]!, out[i * 3 + 1]!, out[i * 3 + 2]!) || 1;
    out[i * 3] = out[i * 3]! / l;
    out[i * 3 + 1] = out[i * 3 + 1]! / l;
    out[i * 3 + 2] = out[i * 3 + 2]! / l;
  }
  return out;
}

/** Landmarks (normalized) → camera-space positions in pixel units (+x right, +y up, +z towards camera). */
export function landmarksToCamera(landmarks: Float32Array, imageW: number, imageH: number, out = new Float32Array(FACE_VERTEX_COUNT * 3)): Float32Array {
  for (let i = 0; i < FACE_VERTEX_COUNT; i++) {
    out[i * 3] = landmarks[i * 3]! * imageW;
    out[i * 3 + 1] = -landmarks[i * 3 + 1]! * imageH;
    out[i * 3 + 2] = -landmarks[i * 3 + 2]! * imageW; // MediaPipe z: smaller = closer, same scale as x
  }
  return out;
}

/**
 * The face-mesh outline (MediaPipe face oval) sits a little *inside* the visible face contour — the
 * relit face looked slightly smaller than the real one, most obvious on the shadow side. Push vertices
 * radially outward from the face centre, weighted by (r / r_max)^4 so only the outer ring moves (eyes,
 * nose, mouth stay put), and push them back in depth by the same amount so the surface keeps rolling
 * off towards the silhouette instead of flattening. `amount` = fraction of the face radius.
 */
export const EXPAND_BINS = 24;

/** Head roll (rad) from the outer eye corners (33, 263), in isotropic pixel space. */
function faceRoll(lm: Float32Array, sx: number, sy: number): number {
  return Math.atan2((lm[263 * 3 + 1]! - lm[33 * 3 + 1]!) * sy, (lm[263 * 3]! - lm[33 * 3]!) * sx);
}

/** Sample a circular per-angle profile (EXPAND_BINS bins, face-aligned angle) with linear interpolation. */
function sampleProfile(profile: Float32Array, ang: number): number {
  const K = profile.length;
  const f = (((ang / (2 * Math.PI)) % 1) + 1) % 1 * K;
  const i0 = Math.floor(f) % K, i1 = (i0 + 1) % K, t = f - Math.floor(f);
  return profile[i0]! * (1 - t) + profile[i1]! * t;
}

export function expandFaceSilhouette(
  landmarks: Float32Array,
  imageW: number,
  imageH: number,
  amount: number,
  out: Float32Array = new Float32Array(FACE_VERTEX_COUNT * 3),
  /** Optional per-direction amounts (face-aligned angle bins), e.g. from estimateFaceExpandProfile. */
  profile?: Float32Array | null,
): Float32Array {
  const n = FACE_VERTEX_COUNT;
  out.set(landmarks.subarray(0, n * 3));
  if (amount <= 0 && !profile) return out;
  const sx = imageW, sy = imageH; // to pixels (isotropic)
  let cx = 0, cy = 0;
  for (let i = 0; i < n; i++) { cx += landmarks[i * 3]! * sx; cy += landmarks[i * 3 + 1]! * sy; }
  cx /= n; cy /= n;
  let rmax = 1e-6;
  for (let i = 0; i < n; i++) rmax = Math.max(rmax, Math.hypot(landmarks[i * 3]! * sx - cx, landmarks[i * 3 + 1]! * sy - cy));
  const roll = profile ? faceRoll(landmarks, sx, sy) : 0;
  for (let i = 0; i < n; i++) {
    const dx = landmarks[i * 3]! * sx - cx, dy = landmarks[i * 3 + 1]! * sy - cy;
    const r = Math.hypot(dx, dy);
    if (r < 1e-6) continue;
    const t = r / rmax;
    const a = profile ? sampleProfile(profile, Math.atan2(dy, dx) - roll) : amount;
    const d = a * rmax * t * t * t * t; // px
    out[i * 3] = landmarks[i * 3]! + (dx / r) * d / sx;
    out[i * 3 + 1] = landmarks[i * 3 + 1]! + (dy / r) * d / sy;
    out[i * 3 + 2] = landmarks[i * 3 + 2]! + (Math.max(0, d) * 1.2) / sx; // z: same scale as x; larger = farther
  }
  return out;
}

/**
 * Measures how far the *real* face (seg face-skin channel) extends beyond the face-mesh outline, per
 * face-aligned direction, in units of the face radius (same units as expandFaceSilhouette's amount).
 *  - short hair / high forehead → skin continues above the mesh → larger expansion up there;
 *  - hair over forehead / temples → skin stops at (or inside) the mesh → little or no expansion.
 * Directions near horizontal (ears are "skin" too) are skipped and filled from neighbours.
 * Returns null if the face is not measurable in this frame.
 *
 * mask: RGBA8 (b = face skin), mw×mh, covering the same frame as the normalised landmarks.
 */
export function estimateFaceExpandProfile(
  mask: Uint8Array,
  mw: number,
  mh: number,
  landmarks: Float32Array,
  imageW: number,
  imageH: number,
  opts: { min?: number; max?: number; earBandDeg?: number } = {},
): Float32Array | null {
  const K = EXPAND_BINS;
  const min = opts.min ?? -0.02, max = opts.max ?? 0.18;
  const earBand = ((opts.earBandDeg ?? 22) * Math.PI) / 180;
  const n = FACE_VERTEX_COUNT, sx = imageW, sy = imageH;
  let cx = 0, cy = 0;
  for (let i = 0; i < n; i++) { cx += landmarks[i * 3]! * sx; cy += landmarks[i * 3 + 1]! * sy; }
  cx /= n; cy /= n;
  const roll = faceRoll(landmarks, sx, sy);
  // Mesh outline radius per bin.
  const rMesh = new Float32Array(K);
  let rmax = 1e-6;
  for (let i = 0; i < n; i++) {
    const dx = landmarks[i * 3]! * sx - cx, dy = landmarks[i * 3 + 1]! * sy - cy;
    const r = Math.hypot(dx, dy);
    rmax = Math.max(rmax, r);
    const f = ((((Math.atan2(dy, dx) - roll) / (2 * Math.PI)) % 1) + 1) % 1;
    const b = Math.floor(f * K) % K;
    rMesh[b] = Math.max(rMesh[b]!, r);
  }
  if (rmax < 20) return null; // face too small to measure
  const skin = (px: number, py: number) => {
    const u = Math.floor((px / sx) * mw), v = Math.floor((py / sy) * mh);
    if (u < 0 || v < 0 || u >= mw || v >= mh) return 0;
    return mask[(v * mw + u) * 4 + 2]! / 255;
  };
  const prof = new Float32Array(K).fill(NaN);
  const step = 0.5 * Math.max(sx / mw, sy / mh, 1); // half a mask texel, in image px
  for (let b = 0; b < K; b++) {
    const ang = ((b + 0.5) / K) * 2 * Math.PI; // face-aligned
    const fromHoriz = Math.asin(Math.min(1, Math.abs(Math.sin(ang))));
    if (fromHoriz < earBand || !rMesh[b]) continue;
    const dirX = Math.cos(ang + roll), dirY = Math.sin(ang + roll);
    const r0 = rMesh[b]! * 0.6, r1 = rMesh[b]! + rmax * (max + 0.1);
    if (skin(cx + dirX * r0, cy + dirY * r0) < 0.5) continue; // no reliable skin inside the face here
    let edge = r0, misses = 0;
    for (let r = r0; r <= r1; r += step) {
      if (skin(cx + dirX * r, cy + dirY * r) >= 0.5) { edge = r; misses = 0; }
      else if (++misses * step > rmax * 0.03) break; // tolerate tiny holes (stubble, glints)
    }
    prof[b] = Math.min(max, Math.max(min, (edge - rMesh[b]!) / rmax));
  }
  // Fill skipped bins by circular linear interpolation between measured neighbours.
  const have = [...prof.keys()].filter((i) => Number.isFinite(prof[i]!));
  if (have.length < K / 4) return null;
  for (let b = 0; b < K; b++) {
    if (Number.isFinite(prof[b]!)) continue;
    let l = b, r = b, dl = 0, dr = 0;
    do { l = (l - 1 + K) % K; dl++; } while (!Number.isFinite(prof[l]!));
    do { r = (r + 1) % K; dr++; } while (!Number.isFinite(prof[r]!));
    prof[b] = (prof[l]! * dr + prof[r]! * dl) / (dl + dr);
  }
  // Light circular smoothing.
  const sm = new Float32Array(K);
  for (let b = 0; b < K; b++) sm[b] = 0.25 * prof[(b - 1 + K) % K]! + 0.5 * prof[b]! + 0.25 * prof[(b + 1) % K]!;
  return sm;
}

/** Per-bin median of several per-frame profiles (robust to blinks, motion blur, hands). */
export function medianProfile(profiles: Float32Array[]): Float32Array | null {
  if (!profiles.length) return null;
  const K = profiles[0]!.length;
  const out = new Float32Array(K);
  const col = new Float32Array(profiles.length);
  for (let b = 0; b < K; b++) {
    profiles.forEach((p, i) => (col[i] = p[b]!));
    const s = Float32Array.from(col).sort();
    out[b] = s[Math.floor(s.length / 2)]!;
  }
  return out;
}

export function buildFaceVertices(
  landmarks: Float32Array, // normalized xyz, ≥ 468 points
  imageW: number,
  imageH: number,
  out: Float32Array = new Float32Array(FACE_VERTEX_COUNT * FACE_VERTEX_STRIDE),
): Float32Array {
  const P = landmarksToCamera(landmarks, imageW, imageH);
  const N = computeVertexNormals(P);
  for (let i = 0; i < FACE_VERTEX_COUNT; i++) {
    const o = i * FACE_VERTEX_STRIDE;
    out[o] = landmarks[i * 3]! * 2 - 1;
    out[o + 1] = 1 - landmarks[i * 3 + 1]! * 2;
    out[o + 2] = Math.min(1, Math.max(0, 0.5 + landmarks[i * 3 + 2]!)); // smaller z = closer = smaller depth
    out[o + 3] = N[i * 3]!;
    out[o + 4] = N[i * 3 + 1]!;
    out[o + 5] = N[i * 3 + 2]!;
    out[o + 6] = CANONICAL_UVS[i * 2]!;
    out[o + 7] = 1 - CANONICAL_UVS[i * 2 + 1]!; // flip to texture space (v down)
  }
  return out;
}

export interface HeadPose {
  yawDeg: number; // + = face turned towards viewer's right (camera +x)
  pitchDeg: number; // + = face tilted up
}

/** Ellipsoid stand-in for the whole head, in image uv / image-height units. */
export interface HeadProxy {
  cx: number;
  cy: number;
  /** Half-axes in image-height units. */
  rx: number;
  ry: number;
  /** In-image roll of the face up-axis (rad, CCW). */
  roll: number;
  yaw: number; // rad
}

/**
 * Fits the head proxy: centred above the face centre so it reaches past the
 * hairline (the face mesh stops mid-forehead), as wide as the cheeks + margin
 * so temples and ears are covered.
 */
export function headProxyFromLandmarks(
  landmarks: Float32Array,
  aspect: number, // image width / height
  yawDeg = 0,
  opts = { up: 0.12, ry: 0.72, rx: 0.56 },
): HeadProxy {
  const P = (i: number): [number, number] => [landmarks[i * 3]! * aspect, landmarks[i * 3 + 1]!]; // height units, y down
  const top = P(10), chin = P(152), r = P(234), l = P(454);
  const ux = top[0] - chin[0];
  const uy = top[1] - chin[1];
  const len = Math.hypot(ux, uy) || 1e-3;
  const width = Math.hypot(l[0] - r[0], l[1] - r[1]);
  const mx = (top[0] + chin[0]) / 2 + (ux / len) * opts.up * len;
  const my = (top[1] + chin[1]) / 2 + (uy / len) * opts.up * len;
  return {
    cx: mx / aspect,
    cy: my,
    rx: opts.rx * width,
    ry: opts.ry * len,
    // Up vector in y-up image coords is (ux, -uy); roll = angle from +y, CCW.
    roll: Math.atan2(-ux, -uy),
    yaw: (yawDeg * Math.PI) / 180,
  };
}

/**
 * Head pose from landmarks, in the same camera axes as our normals:
 * forward = (right-cheek→left-cheek) × (chin→forehead).
 */
export function headPoseFromLandmarks(landmarks: Float32Array, imageW: number, imageH: number): HeadPose {
  const P = landmarksToCamera(landmarks, imageW, imageH);
  const v = (i: number): [number, number, number] => [P[i * 3]!, P[i * 3 + 1]!, P[i * 3 + 2]!];
  const r = v(234), l = v(454), chin = v(152), top = v(10);
  const ax = [l[0] - r[0], l[1] - r[1], l[2] - r[2]];
  const ay = [top[0] - chin[0], top[1] - chin[1], top[2] - chin[2]];
  const f = [ax[1]! * ay[2]! - ax[2]! * ay[1]!, ax[2]! * ay[0]! - ax[0]! * ay[2]!, ax[0]! * ay[1]! - ax[1]! * ay[0]!];
  return {
    yawDeg: (Math.atan2(f[0]!, f[2]!) * 180) / Math.PI,
    pitchDeg: (Math.atan2(f[1]!, Math.hypot(f[0]!, f[2]!)) * 180) / Math.PI,
  };
}

/** Head pose from MediaPipe's 4x4 column-major facial transformation matrix (canonical → camera). */
export function headPoseFromMatrix(m: Float32Array): HeadPose {
  // Face forward axis (canonical +z) in camera space = third column.
  const fx = m[8]!, fy = m[9]!, fz = m[10]!;
  return {
    yawDeg: (Math.atan2(fx, fz) * 180) / Math.PI,
    pitchDeg: (Math.atan2(fy, Math.hypot(fx, fz)) * 180) / Math.PI,
  };
}

/** Lerp weights for the 3 baked UV shadow layers (left / centre / right yaw ≈ -30° / 0° / +30°). */
export function shadowLayerWeights(yawDeg: number, spanDeg = 30): [number, number, number] {
  const t = Math.max(-1, Math.min(1, yawDeg / spanDeg));
  return t < 0 ? [-t, 1 + t, 0] : [0, 1 - t, t];
}
