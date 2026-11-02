import type { LightingRig } from '@relight/shared-schema';
import { targetShading, prepareLights, shEval, DEFAULT_SHADING, type SH9, type ShadingOptions, type Vec3 } from '@relight/relight-shader';
import { exposureGain } from './shFit.ts';

const lum = (v: Vec3) => 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];

/**
 * Returns the room SH to bake against, scaled so the relit face keeps the
 * observed mean luminance (× `exposure`). Without this, the arbitrary albedo
 * scale in the fit would make the result over/under-exposed.
 */
export function exposureNormalizedRoom(
  roomSh: SH9,
  samples: Float32Array,
  rig: LightingRig,
  exposure = 1,
  shading: ShadingOptions = DEFAULT_SHADING,
  minRoom = 0.05,
): { sh: SH9; gain: number } {
  const prepared = prepareLights(rig, shading);
  const t: Vec3 = [0, 0, 0];
  const r: Vec3 = [0, 0, 0];
  // Scale-free first: normalise the room so the divisor floor (minRoom) is meaningful.
  const front = shEval(roomSh, [0, 0, 1]);
  const s0 = 1 / Math.max(lum(front), 1e-6);
  const g = exposureGain(samples, (n) => {
    targetShading(rig, n, shading, prepared, t);
    shEval(roomSh, n, r);
    return lum([
      t[0] / Math.max(r[0] * s0, minRoom),
      t[1] / Math.max(r[1] * s0, minRoom),
      t[2] / Math.max(r[2] * s0, minRoom),
    ]);
  });
  const scale = s0 / (g * exposure);
  const sh = new Float32Array(27);
  for (let i = 0; i < 27; i++) sh[i] = roomSh[i]! * scale;
  return { sh, gain: g * exposure };
}
