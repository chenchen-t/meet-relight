// GPU glue for the LiteRT.js segmentation chain.
// pack:   camera frame (external texture) → frame buffer [1,H,W,4] f32 RGBA in [0,1]
// unpack: mask buffer [1,256,256,4] f32 → rgba8unorm texture (r person, g skin, b face, a hair)

struct PackU { size: vec2u, _pad: vec2u };
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var cam: texture_external;
@group(0) @binding(2) var<storage, read_write> frame: array<vec4f>;
@group(0) @binding(3) var<uniform> pu: PackU;

@compute @workgroup_size(8, 8)
fn pack(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= pu.size.x || gid.y >= pu.size.y) { return; }
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(pu.size);
  frame[gid.y * pu.size.x + gid.x] = vec4f(textureSampleBaseClampToEdge(cam, samp, uv).rgb, 1.0);
}

@group(0) @binding(0) var<storage, read> maskBuf: array<vec4f>;
@group(0) @binding(1) var maskOut: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn unpack(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(maskOut);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  textureStore(maskOut, gid.xy, clamp(maskBuf[gid.y * size.x + gid.x], vec4f(0.0), vec4f(1.0)));
}
