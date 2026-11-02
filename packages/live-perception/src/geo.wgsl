// live-perception GPU geometry: face-mesh rasterization + silhouette ("blob")
// normals for non-face foreground, composed into the u_geo_tex contract.

// ---------------------------------------------------------------- face raster (MRT)
struct VIn {
  @location(0) ndcDepth: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
};
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) normal: vec3f,
  @location(1) uv: vec2f,
};
struct FOut {
  @location(0) geo: vec4f,    // normal xyz, w = 1 (coverage)
  @location(1) faceUv: vec4f, // canonical uv, b = valid
};

@vertex
fn vsFace(v: VIn) -> VOut {
  var o: VOut;
  o.pos = vec4f(v.ndcDepth.xy, v.ndcDepth.z, 1.0);
  o.normal = v.normal;
  o.uv = v.uv;
  return o;
}

@fragment
fn fsFace(v: VOut) -> FOut {
  var o: FOut;
  o.geo = vec4f(normalize(v.normal), 1.0);
  o.faceUv = vec4f(v.uv, 1.0, 1.0);
  return o;
}

// ---------------------------------------------------------------- mask blur (separable box)
struct BlurU {
  dir: vec2i,
  radius: i32,
  channel: i32, // which channel of blurSrc to blur
};
@group(0) @binding(0) var blurSrc: texture_2d<f32>;
@group(0) @binding(1) var blurDst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> bu: BlurU;

@compute @workgroup_size(8, 8)
fn blur(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2i(textureDimensions(blurSrc));
  let p = vec2i(gid.xy);
  if (p.x >= size.x || p.y >= size.y) { return; }
  var acc = 0.0;
  for (var i = -bu.radius; i <= bu.radius; i++) {
    let q = clamp(p + bu.dir * i, vec2i(0), size - 1);
    acc += textureLoad(blurSrc, q, 0)[bu.channel];
  }
  textureStore(blurDst, p, vec4f(acc / f32(2 * bu.radius + 1), 0.0, 0.0, 1.0));
}

// ---------------------------------------------------------------- normal smoothing (separable Gaussian)
// Removes face-mesh facets (nose ridge) and the mesh → proxy/blob seam (jawline) that otherwise show up
// as hard highlight edges. Only the *direction* is smoothed; coverage (|n|) and .w stay per-pixel so the
// relight boundary does not spread onto the background.
@group(0) @binding(0) var snSrc: texture_2d<f32>;
@group(0) @binding(1) var snDst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> snU: BlurU;

@compute @workgroup_size(8, 8)
fn smoothN(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2i(textureDimensions(snSrc));
  let p = vec2i(gid.xy);
  if (p.x >= size.x || p.y >= size.y) { return; }
  let c = textureLoad(snSrc, p, 0);
  let cLen = length(c.xyz);
  if (snU.radius <= 0 || cLen < 1e-3) {
    textureStore(snDst, p, c);
    return;
  }
  let sigma = max(f32(snU.radius) * 0.5, 0.5);
  var acc = vec3f(0.0);
  for (var i = -snU.radius; i <= snU.radius; i++) {
    let q = clamp(p + snU.dir * i, vec2i(0), size - 1);
    let s = textureLoad(snSrc, q, 0).xyz; // |s| = coverage → covered pixels dominate, background ignored
    acc += s * exp(-f32(i * i) / (2.0 * sigma * sigma));
  }
  let aLen = length(acc);
  let n = select(c.xyz / cLen, acc / aLen, aLen > 1e-4);
  textureStore(snDst, p, vec4f(n * cLen, c.w));
}

// ---------------------------------------------------------------- compose
struct ComposeU {
  blobStrength: f32,
  headValid: f32,
  flatten: f32,      // >1 flattens the head proxy front (real faces are flatter than spheres)
  relightHair: f32,  // 0 = exclude seg "hair" class from relighting
  center: vec2f,     // head proxy centre, image uv
  radii: vec2f,      // half-axes in image-height units (x across face, y along face)
  roll: f32,         // radians, in-image rotation of the face's up axis
  yaw: f32,          // radians
  maskShift: vec2f,  // uv shift applied to the (older) seg mask around the head — latency compensation
  edgeSoft: f32,     // coverage feather radius in output pixels (inward-only: no halo on background)
};
@group(0) @binding(0) var cSamp: sampler;
@group(0) @binding(1) var cMask: texture_2d<f32>;     // raw seg mask (r = person, a = hair)
@group(0) @binding(2) var cBlur: texture_2d<f32>;     // blurred person mask
@group(0) @binding(3) var cFace: texture_2d<f32>;     // face raster geo (same size as output)
@group(0) @binding(4) var cOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> cu: ComposeU;
@group(0) @binding(6) var cFaceBlur: texture_2d<f32>; // blurred face coverage (r)
@group(0) @binding(7) var cHeadBlur: texture_2d<f32>; // blurred seg face-skin channel (r)

// Smooth ellipsoid normal for the whole head (forehead, temples, ears).
fn headProxy(uv: vec2f, aspect: f32) -> vec4f {
  let d0 = (uv - cu.center) * vec2f(aspect, 1.0);
  let dimg = vec2f(d0.x, -d0.y); // y up
  let c = cos(cu.roll);
  let s = sin(cu.roll);
  // Rotate into head frame (undo roll).
  let dh = vec2f(c * dimg.x + s * dimg.y, -s * dimg.x + c * dimg.y) / cu.radii;
  let r = length(dh);
  let z = sqrt(max(1.0 - r * r, 0.0)) * cu.flatten;
  var n = normalize(vec3f(dh, max(z, 0.02)));
  // Yaw about head up axis, then roll back into image frame.
  let cy = cos(cu.yaw);
  let sy = sin(cu.yaw);
  n = vec3f(cy * n.x + sy * n.z, n.y, -sy * n.x + cy * n.z);
  n = vec3f(c * n.x - s * n.y, s * n.x + c * n.y, n.z);
  let w = (1.0 - smoothstep(0.85, 1.05, r)) * cu.headValid;
  return vec4f(n, w);
}

@compute @workgroup_size(8, 8)
fn compose(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2i(textureDimensions(cOut));
  let p = vec2i(gid.xy);
  if (p.x >= size.x || p.y >= size.y) { return; }
  let uv = (vec2f(p) + 0.5) / vec2f(size);
  let mt = 1.0 / vec2f(textureDimensions(cBlur));
  let aspect = f32(size.x) / f32(size.y);

  // Head proxy uses the (predicted) current landmarks.
  let head = headProxy(uv, aspect);
  // The seg mask is from an older frame: around the head, sample it where the head *was*.
  let uvm = uv - cu.maskShift * smoothstep(0.0, 0.6, head.w);

  let m = textureSampleLevel(cMask, cSamp, uvm, 0.0);
  // Soft (LiteRT, 256² stretched) masks have a wide probability tail at the silhouette; relighting that
  // tail lights a band of background around the head. Sharpen around the 50% boundary, biased slightly
  // inward. (Hard MediaPipe masks pass through unchanged.)
  let person = smoothstep(0.45, 0.65, m.r);
  let bx = textureSampleLevel(cBlur, cSamp, uvm + vec2f(mt.x, 0.0), 0.0).r - textureSampleLevel(cBlur, cSamp, uvm - vec2f(mt.x, 0.0), 0.0).r;
  let by = textureSampleLevel(cBlur, cSamp, uvm + vec2f(0.0, mt.y), 0.0).r - textureSampleLevel(cBlur, cSamp, uvm - vec2f(0.0, mt.y), 0.0).r;
  // Outward silhouette normal = -gradient (image v points down → camera y flips).
  let blob = normalize(vec3f(-bx * cu.blobStrength, by * cu.blobStrength, 1.0));

  // Soft person / hair fields: disk average over edgeSoft px (min 4 px so the hair→face normal hand-off
  // is always a gradient). pRaw / hRaw are *centred* on the boundary.
  let rad = max(cu.edgeSoft, 4.0) / vec2f(size);
  var pRaw = 0.0;
  var hRaw = 0.0;
  var oRaw = 0.0;
  let N = 16;
  for (var i = 0; i < N; i++) {
    let t = (f32(i) + 0.5) / f32(N);
    let ang = f32(i) * 2.39996323; // golden angle
    let o = vec2f(cos(ang), sin(ang)) * sqrt(t) * rad;
    let mi = textureSampleLevel(cMask, cSamp, uvm + o, 0.0);
    pRaw += smoothstep(0.45, 0.65, mi.r);
    hRaw += smoothstep(0.3, 0.5, mi.a);
    // Occluder in front of the face: body skin (g = body+face skin, b = face skin) → a hand.
    oRaw += smoothstep(0.3, 0.6, mi.g - mi.b);
  }
  pRaw /= f32(N);
  hRaw /= f32(N);
  oRaw /= f32(N);

  // Layer 1: body silhouette normals. Layer 2: head proxy. Layer 3: face mesh. Layer 4: hair.
  // The proxy only applies where segmentation says face skin, so its ellipse never spills onto
  // neck / shoulders / clothes / long hair (the "oversized mask" look).
  let faceSkinBlur = textureSampleLevel(cHeadBlur, cSamp, uvm, 0.0).r;
  let headGate = smoothstep(0.1, 0.5, faceSkinBlur);
  let headW = head.w * headGate;
  // Near the person silhouette, keep some outward tilt from the blob so the head edge rolls off.
  let headN = normalize(head.xyz + (blob - vec3f(0.0, 0.0, 1.0)) * 0.5);
  var n = normalize(mix(blob, headN, headW));

  let f = textureLoad(cFace, p, 0);
  let cov = textureLoad(cFaceBlur, p, 0).r;
  // Face mesh is rasterised from landmarks and knows nothing about what is in front of the face.
  // Where a hand covers it, drop the mesh (its normals, cast shadow, Greenroom samples).
  let occW = smoothstep(0.15, 0.6, oRaw);
  let wFace = smoothstep(0.35, 0.95, cov) * (1.0 - occW);
  // Mesh-edge normals along the cheeks / jaw overshoot to nearly 90° and catch side lights as a thin
  // bright outline; smooth floor on N.z (≈0.35 → ≤ ~70°), negligible change on frontal normals.
  let fn0 = f.xyz;
  let fzc = max(fn0.z, 0.0);
  let faceMeshN = normalize(vec3f(fn0.xy, sqrt(fzc * fzc + 0.1225)));
  let faceN = select(n, faceMeshN, f.w > 0.5);
  // The face mesh continues *under* bangs / fringe; hand off softly to the hair layer there.
  let wFaceN = wFace * (1.0 - smoothstep(0.15, 0.7, hRaw));
  n = normalize(mix(n, faceN, wFaceN));
  // The hand itself: a soft, mostly camera-facing normal (it is roughly a plane in front of the face),
  // tilted a little by the silhouette so it still takes directional light.
  let handN = normalize(mix(blob, vec3f(0.0, 0.0, 1.0), 0.6));
  n = normalize(mix(n, handN, occW * f.w));

  // Hair is a volume of strands, not a smooth shell: give it a broad, boundary-free dome tilt around the
  // head centre (linear in distance, clamped well short of grazing). No ellipse rim, no silhouette band,
  // so relit hair picks up a gentle left/right/top gradient instead of contour lines.
  if (cu.relightHair > 0.5) {
    let d0 = (uv - cu.center) * vec2f(aspect, 1.0);
    let dimg = vec2f(d0.x, -d0.y);
    let dh = select(vec2f(0.0), dimg / max(cu.radii, vec2f(1e-3)), cu.headValid > 0.5);
    let tilt = dh * 0.45;
    let hairN = normalize(vec3f(tilt / max(1.0, length(tilt) / 0.9), 1.0));
    let hairW = smoothstep(0.15, 0.7, hRaw);
    n = normalize(mix(n, hairN, hairW));
  }

  // Relight coverage: person ∪ face mesh, minus hair (unless enabled). Hair wins even
  // inside the face mesh, so bangs/fringe over the forehead keep their original light.
  // Both boundaries are feathered *inward* with the disk average: at the true silhouette / hairline
  // the weight is already 0 (no glow on background or hair), ramping to 1 inside — a gradient, not a cut.
  var pAvg = person;
  var hAvg = smoothstep(0.3, 0.5, m.a);
  if (cu.edgeSoft > 0.5) {
    pAvg = smoothstep(0.5, 1.0, pRaw) * person;
    hAvg = max(smoothstep(0.0, 0.5, hRaw), hAvg);
  }
  let hair = hAvg * (1.0 - cu.relightHair);
  let coverage = max(pAvg, wFace) * (1.0 - hair);
  // Encode coverage in the normal length (contract: |n| = relight weight, 0 = untouched).
  // .w = *soft* face-mesh coverage (feathered, 1 only well inside the mesh), suppressed under hair —
  // mesh-only effects (cast shadow) fade out before the mesh edge instead of cutting a line.
  textureStore(cOut, p, select(vec4f(0.0), vec4f(n * coverage, wFace * f.w * (1.0 - hRaw)), coverage > 0.05));
}
