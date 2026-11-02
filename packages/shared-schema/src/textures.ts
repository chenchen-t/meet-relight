/**
 * Frozen GPU texture contract between live-perception (producer) and
 * relight-shader (consumer). See design doc §4.2. WebGPU formats.
 */
export const TEXTURE_CONTRACT = {
  /** xyz = camera-space unit normal × relight coverage (0 = leave pixel untouched); w = face-mesh coverage. */
  geo: { name: 'u_geo_tex', format: 'rgba16float' as GPUTextureFormat },
  /** r = foreground alpha, g = skin alpha, b = face alpha, a = reserved. */
  mask: { name: 'u_mask_tex', format: 'rgba8unorm' as GPUTextureFormat },
  /** Canonical face UV (rg) for UV-space baked shadows; b = valid flag. */
  faceUv: { name: 'u_face_uv_tex', format: 'rgba16float' as GPUTextureFormat },
  /** 64x64 spherical quotient MatCap: rgb = target/room ratio, indexed by N.xy*0.5+0.5. */
  quotientMatcap: { name: 'u_baked_quotient_matcap', format: 'rgba16float' as GPUTextureFormat, size: 64 },
  /** Canonical-UV cast-shadow maps, 3 layers (left / centre / right yaw). r = key-light visibility. */
  uvShadow: { name: 'u_baked_uv_shadow', format: 'rgba8unorm' as GPUTextureFormat, layers: 3, size: 256 },
} as const;
