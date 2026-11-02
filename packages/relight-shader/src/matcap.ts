/**
 * Spherical Quotient MatCap baker (design doc §2.3).
 *
 *   Q(N) = L_target(N) / L_room(N)
 *
 * stored in a 64×64 texture indexed by matcapUv(N) = (N.x*0.5+0.5, 0.5-N.y*0.5)
 * (WebGPU texture space: v grows downwards, camera-space +y is up).
 * At runtime: relit = original · Q(N) — de-light + add-light in one fetch.
 */
import type { LightingRig } from '@relight/shared-schema';
import { shEval, type SH9 } from './sh.ts';
import { targetShading, prepareLights, DEFAULT_SHADING, type ShadingOptions } from './lighting.ts';
import type { Vec3 } from './vec.ts';

export const MATCAP_SIZE = 64;

/** Room light model: either fitted SH (M3) or a flat constant (no de-lighting). */
export type RoomLight = { kind: 'sh'; sh: SH9 } | { kind: 'flat'; level: number };

export interface BakeOptions {
  size?: number;
  shading?: ShadingOptions;
  /** Lower bound of room shading used as divisor (prevents dark-region noise blow-up). */
  minRoom?: number;
  /** Max ratio (guardrail against amplifying sensor noise in crushed shadows). */
  maxRatio?: number;
  /**
   * Fitted room SH is only constrained by camera-facing samples; for normals with
   * n.z below this, evaluate the room at a normal pulled toward the camera.
   */
  minRoomNz?: number;
  /**
   * Max per-channel deviation of the room colour from its luminance (ratio, ≥1).
   * The fitted room's directional colour is unreliable (camera AWB drift,
   * single-view refits); dividing by it unclamped produces strong colour casts
   * (e.g. a green face). 1 = luminance-only de-lighting.
   */
  maxRoomChroma?: number;
}

/** Pull a normal toward the camera so n.z ≥ minNz (keeps its azimuth). */
export function clampToObserved(n: Vec3, minNz: number): Vec3 {
  if (n[2] >= minNz) return n;
  const xy = Math.hypot(n[0], n[1]) || 1;
  const s = Math.sqrt(1 - minNz * minNz) / xy;
  return [n[0] * s, n[1] * s, minNz];
}

/** Normal for the centre of matcap texel (i, j). Texels outside the unit disc clamp to the rim. */
export function matcapTexelNormal(i: number, j: number, size: number): Vec3 {
  let x = ((i + 0.5) / size) * 2 - 1;
  let y = 1 - ((j + 0.5) / size) * 2;
  const r2 = x * x + y * y;
  if (r2 >= 1) {
    const r = Math.sqrt(r2);
    x /= r;
    y /= r;
    return [x, y, 0];
  }
  return [x, y, Math.sqrt(1 - r2)];
}

export function matcapUv(n: Vec3): [number, number] {
  return [n[0] * 0.5 + 0.5, 0.5 - n[1] * 0.5];
}

export function roomShading(room: RoomLight, n: Vec3, out: Vec3 = [0, 0, 0]): Vec3 {
  if (room.kind === 'flat') {
    out[0] = out[1] = out[2] = room.level;
    return out;
  }
  return shEval(room.sh, n, out);
}

/**
 * Returns RGBA float32 data (size*size*4).
 *   rgb = clamp(L_target / max(L_room, minRoom), maxRatio)
 *   a   = key light's share of target luminance (cast shadows attenuate only this part)
 */
export function bakeQuotientMatcap(rig: LightingRig, room: RoomLight, opts: BakeOptions = {}): Float32Array {
  const size = opts.size ?? MATCAP_SIZE;
  const shading = opts.shading ?? DEFAULT_SHADING;
  const minRoom = opts.minRoom ?? 0.05;
  const maxRatio = opts.maxRatio ?? 4;
  const maxChroma = Math.max(1, opts.maxRoomChroma ?? 1.15);
  const prepared = prepareLights(rig, shading);
  const out = new Float32Array(size * size * 4);
  const t: Vec3 = [0, 0, 0];
  const tk: Vec3 = [0, 0, 0];
  const r: Vec3 = [0, 0, 0];
  const lum = (v: Vec3) => 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
  for (let j = 0; j < size; j++) {
    for (let i = 0; i < size; i++) {
      const n = matcapTexelNormal(i, j, size);
      targetShading(rig, n, shading, prepared, t, tk);
      roomShading(room, room.kind === 'sh' ? clampToObserved(n, opts.minRoomNz ?? 0.45) : n, r);
      const k = (j * size + i) * 4;
      // Limit the room's colour to ±maxRoomChroma around its luminance.
      const rl = Math.max(lum(r), 1e-6);
      for (let c = 0; c < 3; c++) r[c] = rl * Math.min(maxChroma, Math.max(1 / maxChroma, r[c]! / rl));
      let peak = 0;
      for (let c = 0; c < 3; c++) {
        out[k + c] = t[c]! / Math.max(r[c]!, minRoom);
        peak = Math.max(peak, out[k + c]!);
      }
      // Cap brightness without shifting hue (per-channel clipping turns highlights green/magenta).
      if (peak > maxRatio) for (let c = 0; c < 3; c++) out[k + c] = out[k + c]! * (maxRatio / peak);
      out[k + 3] = lum(tk) / Math.max(lum(t), 1e-6);
    }
  }
  return out;
}

/** Bilinear sample of a baked matcap (CPU reference of the GPU fetch; used by tests). */
export function sampleMatcap(data: Float32Array, size: number, n: Vec3): Vec3 {
  const [u, v] = matcapUv(n);
  const fx = Math.min(size - 1, Math.max(0, u * size - 0.5));
  const fy = Math.min(size - 1, Math.max(0, v * size - 0.5));
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const x1 = Math.min(size - 1, x0 + 1);
  const y1 = Math.min(size - 1, y0 + 1);
  const ax = fx - x0;
  const ay = fy - y0;
  const res: Vec3 = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const p = (x: number, y: number) => data[(y * size + x) * 4 + c]!;
    res[c] =
      p(x0, y0) * (1 - ax) * (1 - ay) + p(x1, y0) * ax * (1 - ay) + p(x0, y1) * (1 - ax) * ay + p(x1, y1) * ax * ay;
  }
  return res;
}
