// Greenroom sample gather: on a sparse grid, emit (camera-space normal, linear
// RGB) for clean face-skin pixels. Read back to CPU for the SH9 fit.
// SRC_DECL / SRC_LOAD template tokens (double-brace) are substituted by the host.

struct U {
  grid: vec2u,     // cells
  stride: u32,     // geo pixels per cell
  _pad: u32,
};
@group(0) @binding(0) var samp: sampler;
{{SRC_DECL}}
@group(0) @binding(2) var geoTex: texture_2d<f32>;
@group(0) @binding(3) var maskTex: texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> outBuf: array<vec4f>;
@group(0) @binding(5) var<uniform> u: U;

fn toLinear(c: vec3f) -> vec3f {
  return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= u.grid.x || gid.y >= u.grid.y) { return; }
  let idx = gid.y * u.grid.x + gid.x;
  let gp = vec2i(gid.xy * u.stride + u.stride / 2u);
  let geoSize = vec2f(textureDimensions(geoTex));
  let uv = (vec2f(gp) + 0.5) / geoSize;

  let g = textureLoad(geoTex, gp, 0);
  let m = textureSampleLevel(maskTex, samp, uv, 0.0);
  let srgb = {{SRC_LOAD}};
  let n = g.xyz;
  let isFace = g.w > 0.99;                    // inside the face mesh (not feathered edge)
  let isSkin = m.g > 0.7 && m.a < 0.2;        // seg: skin, not hair (works for soft or hard masks)
  let notClipped = max(srgb.r, max(srgb.g, srgb.b)) < 0.97 && min(srgb.r, min(srgb.g, srgb.b)) > 0.02;
  let ok = isFace && isSkin && notClipped && length(n) > 0.5;
  outBuf[idx * 2u] = vec4f(n, select(0.0, 1.0, ok));
  outBuf[idx * 2u + 1u] = vec4f(toLinear(srgb), 0.0);
}
