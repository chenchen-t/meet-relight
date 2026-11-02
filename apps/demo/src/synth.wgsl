// Synthetic "subject under a known room light" generator for the M1 playground.
// Produces exactly the textures live-perception will produce in M2:
//   colorOut : sRGB camera-like frame = albedo · L_room(N)
//   geoOut   : camera-space normal (xyz) + depth (w)
//   maskOut  : r person, g skin, b face, a hair
// Because L_room is the same SH9 the baker divides by, relighting is exact up to
// MatCap quantisation — a ground-truth sandbox for the shader.

struct U {
  size: vec2f,
  _pad: vec2f,
  sh: array<vec4f, 9>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var colorOut: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var geoOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var maskOut: texture_storage_2d<rgba8unorm, write>;

fn shEval(n: vec3f) -> vec3f {
  let x = n.x; let y = n.y; let z = n.z;
  var r = u.sh[0].xyz * 0.282095;
  r += u.sh[1].xyz * (0.488603 * y);
  r += u.sh[2].xyz * (0.488603 * z);
  r += u.sh[3].xyz * (0.488603 * x);
  r += u.sh[4].xyz * (1.092548 * x * y);
  r += u.sh[5].xyz * (1.092548 * y * z);
  r += u.sh[6].xyz * (0.315392 * (3.0 * z * z - 1.0));
  r += u.sh[7].xyz * (1.092548 * x * z);
  r += u.sh[8].xyz * (0.546274 * (x * x - y * y));
  return max(r, vec3f(0.0));
}

fn hash(p: vec2f) -> f32 {
  let h = dot(p, vec2f(127.1, 311.7));
  return fract(sin(h) * 43758.5453);
}
fn vnoise(p: vec2f) -> f32 {
  let i = floor(p); let f = fract(p);
  let s = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2f(1, 0)), s.x), mix(hash(i + vec2f(0, 1)), hash(i + vec2f(1, 1)), s.x), s.y);
}

// Front-surface hit of an axis-aligned ellipsoid (orthographic camera looking down -z).
// Returns (z, nx, ny, nz) or z = -1e9 on miss.
fn ellipsoid(p: vec2f, c: vec3f, r: vec3f) -> vec4f {
  let d = (p - c.xy) / r.xy;
  let k = 1.0 - dot(d, d);
  if (k <= 0.0) { return vec4f(-1e9, 0, 0, 0); }
  let zl = sqrt(k);
  let z = c.z + zl * r.z;
  let n = normalize(vec3f(d.x / r.x, d.y / r.y, zl / r.z));
  return vec4f(z, n);
}

fn toSrgb(c: vec3f) -> vec3f {
  let x = clamp(c, vec3f(0.0), vec3f(1.0));
  return select(1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3f(0.0031308));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (f32(gid.x) >= u.size.x || f32(gid.y) >= u.size.y) { return; }
  let H = u.size.y;
  let px = vec2f(gid.xy) + 0.5;
  // Scene coords: unit = image height, origin at centre, +y up.
  let p = vec2f((px.x - 0.5 * u.size.x) / H, (0.5 * H - px.y) / H);

  let head = ellipsoid(p, vec3f(0.0, 0.06, 0.0), vec3f(0.17, 0.23, 0.18));
  let nose = ellipsoid(p, vec3f(0.0, 0.02, 0.16), vec3f(0.025, 0.06, 0.04));
  let body = ellipsoid(p, vec3f(0.0, -0.52, -0.1), vec3f(0.42, 0.26, 0.2));
  let hair = ellipsoid(p, vec3f(0.0, 0.14, -0.02), vec3f(0.185, 0.19, 0.17));

  var hit = vec4f(-1e9, 0, 0, 0);
  var kind = 0u; // 0 bg, 1 skin, 2 shirt, 3 hair
  if (body.x > hit.x) { hit = body; kind = 2u; }
  if (head.x > hit.x) { hit = head; kind = 1u; }
  // Hair cap: only on the upper part of the head.
  if (hair.x > hit.x - 0.002 && p.y > 0.12 + 0.05 * abs(p.x) / 0.18) { hit = hair; kind = 3u; }
  if (nose.x > hit.x) { hit = nose; kind = 1u; }

  if (kind == 0u) {
    // Back wall, lit by room ambient only (not relit: mask = 0).
    let wall = vec3f(0.32, 0.31, 0.30) * shEval(vec3f(0.0, 0.0, 1.0)) * (0.92 + 0.08 * vnoise(px * 0.01));
    textureStore(colorOut, gid.xy, vec4f(toSrgb(wall), 1.0));
    textureStore(geoOut, gid.xy, vec4f(0.0));
    textureStore(maskOut, gid.xy, vec4f(0.0));
    return;
  }

  let n = hit.yzw;
  var albedo: vec3f;
  var skin = 0.0;
  var face = 0.0;
  var hairM = 0.0;
  if (kind == 1u) {
    skin = 1.0; face = 1.0;
    albedo = vec3f(0.62, 0.40, 0.32);
    // pores / micro texture (the detail the quotient multiply must preserve)
    let pores = vnoise(px * 0.9) * 0.6 + vnoise(px * 0.23) * 0.4;
    albedo *= 0.9 + 0.2 * pores;
    // eyes + brows + lips (albedo features, no geometry)
    let q = (p - vec2f(0.0, 0.06)) / vec2f(0.17, 0.23);
    let eyeL = length((q - vec2f(-0.38, 0.12)) / vec2f(0.17, 0.07));
    let eyeR = length((q - vec2f(0.38, 0.12)) / vec2f(0.17, 0.07));
    let iris = min(length((q - vec2f(-0.38, 0.12)) / vec2f(0.06, 0.06)), length((q - vec2f(0.38, 0.12)) / vec2f(0.06, 0.06)));
    if (min(eyeL, eyeR) < 1.0) { albedo = vec3f(0.75, 0.72, 0.70); skin = 0.0; }
    if (iris < 1.0) { albedo = vec3f(0.06, 0.04, 0.03); }
    let browL = length((q - vec2f(-0.38, 0.33)) / vec2f(0.22, 0.035));
    let browR = length((q - vec2f(0.38, 0.33)) / vec2f(0.22, 0.035));
    if (min(browL, browR) < 1.0) { albedo = vec3f(0.08, 0.05, 0.04); }
    let lips = length((q - vec2f(0.0, -0.45)) / vec2f(0.28, 0.07));
    if (lips < 1.0) { albedo = vec3f(0.55, 0.18, 0.18); }
  } else if (kind == 2u) {
    albedo = vec3f(0.10, 0.16, 0.30) * (0.85 + 0.3 * vnoise(px * 0.5));
  } else {
    hairM = 1.0;
    albedo = vec3f(0.05, 0.035, 0.025) * (0.6 + 0.8 * vnoise(vec2f(px.x * 0.6, px.y * 0.05)));
  }

  let lit = albedo * shEval(n);
  textureStore(colorOut, gid.xy, vec4f(toSrgb(lit), 1.0));
  textureStore(geoOut, gid.xy, vec4f(n, select(0.0, 1.0, kind == 1u))); // w = face coverage (contract)
  textureStore(maskOut, gid.xy, vec4f(1.0, skin, face, hairM));
}
