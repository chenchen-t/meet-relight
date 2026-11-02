// Matte refinement: fast guided filter (He et al.) on the 256² segmentation person channel, guided by
// the camera frame at geo resolution, + motion-adaptive temporal smoothing.
//
//  pack   : I = luma(camera), p = mask.r            → (I, p, I·p, I·I)
//  box    : separable box mean (4 channels)          → means
//  coef   : a = cov(I,p) / (var(I) + ε), b = μp − a μI → (a, b)
//  box    : box mean of (a, b)
//  apply  : q = ā·I + b̄ ; temporal: mix(prev, q, k(|q − prev|))

struct PackU { eps: f32, temporal: f32, _p0: f32, _p1: f32 };

// ---------------------------------------------------------------- pack
@group(0) @binding(0) var pSamp: sampler;
@group(0) @binding(1) var pCam: texture_external;
@group(0) @binding(2) var pMask: texture_2d<f32>;
@group(0) @binding(3) var pOut: texture_storage_2d<rgba16float, write>;

fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.299, 0.587, 0.114)); }

@compute @workgroup_size(8, 8)
fn pack(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(pOut);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(size);
  let I = luma(textureSampleBaseClampToEdge(pCam, pSamp, uv).rgb);
  let p = textureSampleLevel(pMask, pSamp, uv, 0.0).r;
  textureStore(pOut, gid.xy, vec4f(I, p, I * p, I * I));
}

// ---------------------------------------------------------------- separable box mean (all 4 channels)
struct BoxU { dir: vec2i, radius: i32, _pad: i32 };
@group(0) @binding(0) var bSrc: texture_2d<f32>;
@group(0) @binding(1) var bDst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> bu: BoxU;

@compute @workgroup_size(8, 8)
fn box(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2i(textureDimensions(bSrc));
  let p = vec2i(gid.xy);
  if (p.x >= size.x || p.y >= size.y) { return; }
  var acc = vec4f(0.0);
  for (var i = -bu.radius; i <= bu.radius; i++) {
    acc += textureLoad(bSrc, clamp(p + bu.dir * i, vec2i(0), size - 1), 0);
  }
  textureStore(bDst, p, acc / f32(2 * bu.radius + 1));
}

// ---------------------------------------------------------------- coefficients
@group(0) @binding(0) var cMeans: texture_2d<f32>;
@group(0) @binding(1) var cOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> cu: PackU;

@compute @workgroup_size(8, 8)
fn coef(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(cOut);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let m = textureLoad(cMeans, gid.xy, 0); // μI, μp, μIp, μII
  let varI = max(m.w - m.x * m.x, 0.0);
  let cov = m.z - m.x * m.y;
  let a = cov / (varI + cu.eps);
  let b = m.y - a * m.x;
  textureStore(cOut, gid.xy, vec4f(a, b, 0.0, 0.0));
}

// ---------------------------------------------------------------- apply + temporal
@group(0) @binding(0) var aAB: texture_2d<f32>;     // box-averaged (a, b)
@group(0) @binding(1) var aPacked: texture_2d<f32>; // I in .x
@group(0) @binding(2) var aPrev: texture_2d<f32>;
@group(0) @binding(3) var aOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> au: PackU;

@compute @workgroup_size(8, 8)
fn apply(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(aOut);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let ab = textureLoad(aAB, gid.xy, 0).xy;
  let I = textureLoad(aPacked, gid.xy, 0).x;
  let raw = textureLoad(aPacked, gid.xy, 0).y;
  var q = clamp(ab.x * I + ab.y, 0.0, 1.0);
  // Guided filter can't invent a matte where seg says "certainly not person" / "certainly person".
  q = select(q, raw, raw < 0.02 || raw > 0.98);
  // Motion-adaptive temporal smoothing: still edges stop flickering, moving edges follow immediately.
  let prev = textureLoad(aPrev, gid.xy, 0).x;
  let k = select(1.0, clamp(au.temporal + abs(q - prev) * 4.0, au.temporal, 1.0), au.temporal < 0.999);
  let outA = mix(prev, q, k);
  textureStore(aOut, gid.xy, vec4f(outA, 0.0, 0.0, 1.0));
}
