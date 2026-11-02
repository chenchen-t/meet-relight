/**
 * Frozen data contract shared by all packages (design doc §4.1).
 * Keep in sync with ../target_profile.schema.json — a unit test enforces the
 * example profile validates against the schema.
 */

export type SubjectType =
  | 'portrait'
  | 'object_diffuse'
  | 'object_specular'
  | 'object_transparent';

export type LightRole = 'key' | 'fill' | 'rim';
export type LightShape = 'octagon' | 'circle_ring' | 'rect_softbox' | 'stripbox' | 'point';
export type LightExecution = 'physical_lamp' | 'virtual_shader';
export type RGB = [number, number, number];

export interface Light {
  id: string;
  role: LightRole;
  /** 0 = camera axis; negative = subject's right (viewer's left). */
  azimuth_deg: number;
  elevation_deg: number;
  clock_direction?: string;
  intensity: number;
  color_rgb: RGB;
  kelvin?: number;
  /** 0 = bare bulb hard edge, 1 = very large soft source. */
  softness: number;
  shape: LightShape;
  assigned_execution: LightExecution;
}

export interface LightingRig {
  ambient_rgb: RGB;
  key_to_fill_ratio?: number;
  lights: Light[];
}

export interface SplitToning {
  shadow_tint_rgb: RGB;
  highlight_tint_rgb: RGB;
  balance: number;
}

export interface ColorDNA {
  lab_mean: RGB;
  lab_std: RGB;
  split_toning?: SplitToning;
  luma_quantiles?: [number, number, number, number, number];
  fade_amount?: number;
  grain_amount?: number;
}

export interface TargetProfile {
  subject_type: SubjectType;
  lighting_rig: LightingRig;
  color_dna: ColorDNA;
}
