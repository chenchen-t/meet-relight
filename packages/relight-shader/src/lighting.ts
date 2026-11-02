/**
 * Analytic target-light model evaluated per surface normal (camera space).
 * Implements the design-doc §2.1 parametric model:
 *   S(N) ≈ I_ambient + Σ_k I_k · c_k · max(N·ω_k, 0)^γ_k
 * extended with
 *   - softness → wrap lighting (large soft sources light past the terminator)
 *   - skin SSS → Pre-Integrated Skin Shading LUT (d'Eon profile, see sss.ts)
 */
import { lightDirection, type LightingRig, type Light } from '@relight/shared-schema';
import { dot, type Vec3 } from './vec.ts';
import { preintegratedSkinLut, sampleSssLut, type SssLut } from './sss.ts';

export interface ShadingOptions {
  /** 0 = no SSS (objects), 1 = skin (curvature radius ≈ 6 mm), in between = flatter skin. */
  sss: number;
  /** Exponent γ applied to the wrapped cosine (1 = Lambert). Ignored when SSS is active. */
  gamma: number;
}

export const DEFAULT_SHADING: ShadingOptions = { sss: 1, gamma: 1 };

interface PreparedLight {
  dir: Vec3;
  rgb: Vec3;
  wrap: number;
  isKey: boolean;
}

export interface PreparedRig {
  lights: PreparedLight[];
  lut: SssLut | null;
}

export function prepareLights(rig: LightingRig, opts: ShadingOptions = DEFAULT_SHADING): PreparedRig {
  const keyIdx = Math.max(0, rig.lights.findIndex((l) => l.role === 'key'));
  return {
    lights: rig.lights.map((l: Light, i) => ({
      dir: lightDirection(l.azimuth_deg, l.elevation_deg),
      rgb: [l.color_rgb[0] * l.intensity, l.color_rgb[1] * l.intensity, l.color_rgb[2] * l.intensity],
      // Hard bare bulb → pure Lambert; big octabox → light wraps ~half way past terminator.
      wrap: 0.5 * l.softness,
      isKey: i === keyIdx,
    })),
    lut: opts.sss > 0.01 ? preintegratedSkinLut(6 / opts.sss) : null,
  };
}

const tmp: Vec3 = [0, 0, 0];

/**
 * Evaluates total target shading (and optionally the key light's share,
 * used to restrict baked cast shadows to the key light only).
 */
export function targetShading(
  rig: LightingRig,
  n: Vec3,
  opts: ShadingOptions = DEFAULT_SHADING,
  prepared: PreparedRig = prepareLights(rig, opts),
  out: Vec3 = [0, 0, 0],
  outKey?: Vec3,
): Vec3 {
  out[0] = rig.ambient_rgb[0];
  out[1] = rig.ambient_rgb[1];
  out[2] = rig.ambient_rgb[2];
  if (outKey) outKey[0] = outKey[1] = outKey[2] = 0;
  for (const l of prepared.lights) {
    const ndl = dot(n, l.dir);
    const w = (ndl + l.wrap) / (1 + l.wrap); // softness: wrap in cosine space, still in [-1, 1]
    if (prepared.lut) {
      sampleSssLut(prepared.lut, w, tmp);
    } else {
      const v = Math.max(w, 0);
      tmp[0] = tmp[1] = tmp[2] = opts.gamma === 1 ? v : Math.pow(v, opts.gamma);
    }
    for (let c = 0; c < 3; c++) {
      const e = l.rgb[c]! * tmp[c]!;
      out[c] = out[c]! + e;
      if (outKey && l.isKey) outKey[c] = outKey[c]! + e;
    }
  }
  return out;
}
