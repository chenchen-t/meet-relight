import Ajv from 'ajv';
import schema from '../target_profile.schema.json';
import type { TargetProfile, Light } from './types.ts';

export * from './types.ts';
export * from './textures.ts';
export { schema as targetProfileSchema };

const ajv = new Ajv({ allErrors: true, strict: false });
const validateFn = ajv.compile<TargetProfile>(schema);

export interface ValidationResult {
  ok: boolean;
  errors: string[];
}

export function validateTargetProfile(value: unknown): ValidationResult {
  const ok = validateFn(value);
  return {
    ok,
    errors: ok
      ? []
      : (validateFn.errors ?? []).map((e) => `${e.instancePath || '/'} ${e.message ?? ''}`.trim()),
  };
}

/**
 * Camera-space convention used by every package:
 *   +x = viewer's right, +y = up, +z = towards the camera.
 * azimuth 0 = light on camera axis; negative azimuth = subject's right = viewer's left.
 */
export function lightDirection(azimuthDeg: number, elevationDeg: number): [number, number, number] {
  const az = (azimuthDeg * Math.PI) / 180;
  const el = (elevationDeg * Math.PI) / 180;
  return [Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)];
}

/** Map an azimuth to a photographer's top-down clock direction (12 = camera, 6 = behind subject). */
export function azimuthToClock(azimuthDeg: number): string {
  // Clock seen from above, subject at centre, camera at 12. Viewer's right = 3 o'clock.
  let hours = (azimuthDeg / 30 + 12) % 12;
  if (hours < 0) hours += 12;
  const halfSteps = Math.round(hours * 2) % 24;
  const h = Math.floor(halfSteps / 2) || 12;
  return `${h}:${halfSteps % 2 ? '30' : '00'}`;
}

export function lightDir(light: Light): [number, number, number] {
  return lightDirection(light.azimuth_deg, light.elevation_deg);
}

/** Example profile copied from design doc §4.1 (enums resolved). */
export const EXAMPLE_PROFILE: TargetProfile = {
  subject_type: 'portrait',
  lighting_rig: {
    ambient_rgb: [0.15, 0.15, 0.18],
    key_to_fill_ratio: 3.2,
    lights: [
      {
        id: 'key_light',
        role: 'key',
        azimuth_deg: -45,
        elevation_deg: 30,
        clock_direction: '10:30',
        intensity: 0.85,
        color_rgb: [1.0, 0.94, 0.88],
        kelvin: 5200,
        softness: 0.75,
        shape: 'octagon',
        assigned_execution: 'virtual_shader',
      },
    ],
  },
  color_dna: {
    lab_mean: [62.5, 4.2, 8.1],
    lab_std: [18.4, 6.1, 9.5],
    split_toning: {
      shadow_tint_rgb: [0.08, 0.12, 0.18],
      highlight_tint_rgb: [1.0, 0.95, 0.85],
      balance: 0.1,
    },
    luma_quantiles: [5, 22, 50, 78, 95],
    fade_amount: 0.06,
    grain_amount: 0.12,
  },
};
