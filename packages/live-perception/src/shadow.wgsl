// Canonical-UV cast-shadow baker (design doc §2.3 table row 3).
// Pass 1 rasterizes the canonical face mesh in UV space → per-texel canonical
// position + normal. Pass 2 ray-traces every texel against all 898 triangles
// toward the key light (jittered over a cone for soft penumbrae), for 3 head-yaw
// layers. Output: key-light visibility, sampled at runtime via face UV.

// ---------------------------------------------------------------- pass 1: UV raster
struct RIn {
  @location(0) pos: vec3f,
  @location(1) nrm: vec3f,
  @location(2) uv: vec2f,
};
struct ROut {
  @builtin(position) clip: vec4f,
  @location(0) pos: vec3f,
  @location(1) nrm: vec3f,
};
struct RFrag {
  @location(0) pos: vec4f,
  @location(1) nrm: vec4f,
};

@vertex
fn vsUv(v: RIn) -> ROut {
  var o: ROut;
  // uv in texture space (v down) → clip space.
  o.clip = vec4f(v.uv.x * 2.0 - 1.0, 1.0 - v.uv.y * 2.0, 0.5, 1.0);
  o.pos = v.pos;
  o.nrm = v.nrm;
  return o;
}

@fragment
fn fsUv(v: ROut) -> RFrag {
  var o: RFrag;
  o.pos = vec4f(v.pos, 1.0);
  o.nrm = vec4f(normalize(v.nrm), 1.0);
  return o;
}

// ---------------------------------------------------------------- pass 2: ray trace
struct TU {
  lightDirs: array<vec4f, 3>, // light direction in head frame per yaw layer (xyz)
  coneRad: f32,
  samples: u32,
  triCount: u32,
  _pad: u32,
};
@group(0) @binding(0) var posTex: texture_2d<f32>;
@group(0) @binding(1) var nrmTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> verts: array<f32>;
@group(0) @binding(3) var<storage, read> tris: array<u32>;
@group(0) @binding(4) var outTex: texture_storage_2d_array<rgba8unorm, write>;
@group(0) @binding(5) var<uniform> tu: TU;

fn vtx(i: u32) -> vec3f {
  return vec3f(verts[i * 3u], verts[i * 3u + 1u], verts[i * 3u + 2u]);
}

// Möller–Trumbore, returns true on hit with t > eps.
fn hit(o: vec3f, d: vec3f, a: vec3f, b: vec3f, c: vec3f) -> bool {
  let e1 = b - a;
  let e2 = c - a;
  let p = cross(d, e2);
  let det = dot(e1, p);
  if (abs(det) < 1e-7) { return false; }
  let inv = 1.0 / det;
  let s = o - a;
  let uu = dot(s, p) * inv;
  if (uu < 0.0 || uu > 1.0) { return false; }
  let q = cross(s, e1);
  let vv = dot(d, q) * inv;
  if (vv < 0.0 || uu + vv > 1.0) { return false; }
  return dot(e2, q) * inv > 0.02;
}

@compute @workgroup_size(8, 8, 1)
fn trace(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(posTex);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let layer = gid.z;
  let P = textureLoad(posTex, gid.xy, 0);
  let Nn = textureLoad(nrmTex, gid.xy, 0);
  if (P.w < 0.5) {
    textureStore(outTex, gid.xy, layer, vec4f(1.0));
    return;
  }
  let N = normalize(Nn.xyz);
  let L = normalize(tu.lightDirs[layer].xyz);
  // Tangent basis around L for cone jitter.
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(L.y) > 0.9);
  let T = normalize(cross(up, L));
  let B = cross(L, T);
  let origin = P.xyz + N * 0.08; // cm bias
  var lit = 0.0;
  for (var s = 0u; s < tu.samples; s++) {
    // Fibonacci disk sample.
    let fr = (f32(s) + 0.5) / f32(tu.samples);
    let r = sqrt(fr) * tu.coneRad;
    let phi = f32(s) * 2.39996323;
    let d = normalize(L + T * (r * cos(phi)) + B * (r * sin(phi)));
    if (dot(N, d) <= 0.0) { lit += 1.0; continue; } // attached shadow is the MatCap's job
    var blocked = false;
    for (var t = 0u; t < tu.triCount && !blocked; t++) {
      blocked = hit(origin, d, vtx(tris[t * 3u]), vtx(tris[t * 3u + 1u]), vtx(tris[t * 3u + 2u]));
    }
    lit += select(1.0, 0.0, blocked);
  }
  let v = lit / f32(tu.samples);
  textureStore(outTex, gid.xy, layer, vec4f(v, v, v, 1.0));
}
