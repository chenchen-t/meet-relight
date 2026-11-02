// ROI warp for the face landmark model: rotated square crop of the camera
// frame → [1,256,256,3] f32 RGB in [0,1] (tightly packed NHWC).
// crop pixel (u,v) ∈ [0,1]² ↦ frame pixel  c + R(angle)·((u,v) − ½)·size   (y down)

struct WarpU {
  center: vec2f,   // frame pixels
  size: f32,       // frame pixels
  angle: f32,      // radians
  frame: vec2f,    // frame size in pixels (coordinate space of center/size)
  out: f32,        // output side (256)
  _pad: f32,
};
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var cam: texture_external;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> wu: WarpU;

@compute @workgroup_size(8, 8)
fn warp(@builtin(global_invocation_id) gid: vec3u) {
  let n = u32(wu.out);
  if (gid.x >= n || gid.y >= n) { return; }
  let o = (vec2f(gid.xy) + 0.5) / wu.out - 0.5;
  let c = cos(wu.angle);
  let s = sin(wu.angle);
  let p = wu.center + vec2f(c * o.x - s * o.y, s * o.x + c * o.y) * wu.size;
  let uv = p / wu.frame;
  var rgb = textureSampleBaseClampToEdge(cam, samp, uv).rgb;
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { rgb = vec3f(0.0); }
  let i = (gid.y * n + gid.x) * 3u;
  dst[i] = rgb.r;
  dst[i + 1u] = rgb.g;
  dst[i + 2u] = rgb.b;
}
