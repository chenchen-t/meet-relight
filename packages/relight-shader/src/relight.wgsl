// Meet-Relight online relight kernel — WGSL port of design doc §3.1.
// Quotient MatCap (de-light + add-light + SSS in ONE fetch) + UV cast shadow
// + Blinn-Phong spec + Fresnel rim. All lighting math in linear space.
//
// The SRC_DECL / SRC_SAMPLE template tokens (double-brace) are substituted by RelightRenderer so the same
// kernel serves camera frames (texture_external) and still images (texture_2d).

struct Uniforms {
  ccm: mat3x3f,          // offset 0   (3 x vec4 padded)
  keyDir: vec3f,         // 48
  roughness: f32,        // 60
  rimColor: vec3f,       // 64  (linear RGB * intensity)
  specStrength: f32,     // 76
  shadowW: vec3f,        // 80  lerp weights of the 3 UV shadow layers (left/centre/right)
  mirror: f32,           // 92
  keyColor: vec3f,       // 96
  mode: f32,             // 108  0 relit | 1 original | 2 split | 3 normals | 4 ratio | 5 mask
  splitX: f32,           // 112
  strength: f32,         // 116  global blend 0..1
  shadowStrength: f32,   // 120
  highlightSoft: f32,    // 124  0..1 shoulder on added highlights + broader spec lobe
  rimDir: vec3f,         // 128  rim light direction (camera space)
  edgeFade: f32,         // 140  0..1 fade relighting toward the original at grazing normals
  bg: vec4f,             // 144  x: 1 = replace background, y: bg image aspect (w/h), z: light-wrap strength
};

@group(0) @binding(0) var samp: sampler;
{{SRC_DECL}}
@group(0) @binding(2) var geoTex: texture_2d<f32>;      // xyz = normal * coverage (camera space), w = face-mesh coverage
@group(0) @binding(3) var maskTex: texture_2d<f32>;     // r fg, g skin, b face, a hair
@group(0) @binding(4) var matcapTex: texture_2d<f32>;   // 64x64 quotient
@group(0) @binding(5) var shadowTex: texture_2d_array<f32>; // canonical-UV cast shadow, 3 layers
@group(0) @binding(6) var faceUvTex: texture_2d<f32>;   // rg canonical uv, b valid
@group(0) @binding(7) var<uniform> u: Uniforms;
@group(0) @binding(8) var bgTex: texture_2d<f32>;       // virtual background (sRGB texture → linear on sample)
@group(0) @binding(9) var matteTex: texture_2d<f32>;    // refined person matte (.r), valid when bg.w = 1

struct VO {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VO {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VO;
  o.pos = vec4f(p[i], 0.0, 1.0);
  o.uv = vec2f(p[i].x * 0.5 + 0.5, 0.5 - p[i].y * 0.5);
  return o;
}

fn toLinear(c: vec3f) -> vec3f {
  return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}
fn toSrgb(c: vec3f) -> vec3f {
  let x = max(c, vec3f(0.0));
  return select(1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3f(0.0031308));
}

fn srcAt(p: vec2f) -> vec3f {
  let uv = p;
  return {{SRC_SAMPLE}};
}

@fragment
fn fs(in: VO) -> @location(0) vec4f {
  let uv = select(in.uv, vec2f(1.0 - in.uv.x, in.uv.y), u.mirror > 0.5);
  let origSrgb = srcAt(uv);
  let orig = toLinear(origSrgb);

  let geo = textureSampleLevel(geoTex, samp, uv, 0.0);
  let mask = textureSampleLevel(maskTex, samp, uv, 0.0);
  let nLen = length(geo.xyz);
  let N = select(vec3f(0.0, 0.0, 1.0), geo.xyz / max(nLen, 1e-4), nLen > 1e-3);
  // Relight only where we have both a person mask and a valid normal; ease off at
  // grazing normals (silhouette), where geometry and the room fit are least reliable.
  let edge = mix(1.0, smoothstep(0.05, 0.45, N.z), u.edgeFade);
  // Contract: |geo.xyz| = relight coverage (person minus excluded regions, soft edges).
  let cover = clamp(nLen, 0.0, 1.0);
  let alpha = cover * u.strength * edge;

  let V = vec3f(0.0, 0.0, 1.0);
  let L = normalize(u.keyDir);
  let H = normalize(L + V);

  // 1. MatCap UV (WebGPU texture space: v down, camera +y up).
  let matcapUv = vec2f(N.x * 0.5 + 0.5, 0.5 - N.y * 0.5);
  // 2. ONE fetch: de-light + add-light + SSS (rgb); a = key light's share of target.
  let mc = textureSampleLevel(matcapTex, samp, matcapUv, 0.0);
  let ratio = mc.rgb;

  // 3. Canonical-UV baked cast shadow, lerped across 3 yaw layers.
  let fuv = textureSampleLevel(faceUvTex, samp, uv, 0.0);
  var shadow = 1.0;
  if (fuv.b > 0.5) {
    let s0 = textureSampleLevel(shadowTex, samp, fuv.rg, 0, 0.0).r;
    let s1 = textureSampleLevel(shadowTex, samp, fuv.rg, 1, 0.0).r;
    let s2 = textureSampleLevel(shadowTex, samp, fuv.rg, 2, 0.0).r;
    shadow = dot(vec3f(s0, s1, s2), u.shadowW);
    // The mesh (and its UV shadow) continues under bangs — fade it out over hair.
    shadow = mix(shadow, 1.0, smoothstep(0.3, 0.6, mask.a));
  }
  // The cast-shadow map only exists on the face mesh; outside it shadow = 1. Two things made that
  // boundary visible as a hard line down the turned-away cheek:
  //  - the binary mesh edge → fade the shadow out with the *soft* mesh coverage (geo.w);
  //  - rays from surfaces already facing away from the key hit the head itself, so the map also
  //    darkens what the matcap terminator darkens → only apply it where N·L > 0 (cast shadows are
  //    for the *lit* side: nose, brow, cheek under the eye socket).
  let shadowW = smoothstep(0.45, 0.95, geo.w) * smoothstep(0.0, 0.35, dot(N, L));
  shadow = mix(1.0, shadow, shadowW);

  // 4. Blinn-Phong spec + Fresnel rim. highlightSoft caps the exponent (≤ 2^10 → ≤ 2^6) so the
  // highlight on the nose / cheekbones is a broad sheen rather than a tight hot spot on mesh facets.
  let shininess = pow(2.0, (1.0 - u.roughness) * mix(10.0, 6.0, u.highlightSoft));
  let spec = pow(max(dot(N, H), 0.0), shininess) * (1.0 - u.roughness) * max(dot(N, L), 0.0);
  // Rim: Fresnel at grazing angles, but only on the silhouette side facing the rim light.
  let rimXY = u.rimDir.xy;
  let nXY = N.xy / max(length(N.xy), 1e-4);
  let rimSide = select(1.0, smoothstep(-0.1, 0.7, dot(nXY, rimXY / max(length(rimXY), 1e-4))), length(rimXY) > 0.15);
  // A rim light is only visible where the surface meets the *background*. The face mesh also has
  // grazing normals along cheeks / jaw where the neighbour is hair or neck (still the person), which
  // drew a bright outline around the face. Probe the person mask a little way outward along N.
  let gdim = vec2f(textureDimensions(geoTex));
  let outUv = vec2f(nXY.x, -nXY.y) * vec2f(gdim.y / gdim.x, 1.0);
  let bg1 = 1.0 - textureSampleLevel(maskTex, samp, uv + outUv * 0.015, 0.0).r;
  let bg2 = 1.0 - textureSampleLevel(maskTex, samp, uv + outUv * 0.035, 0.0).r;
  let silhouette = smoothstep(0.3, 0.8, max(bg1, bg2));
  let fresnel = pow(1.0 - max(dot(N, V), 0.0), 4.0) * rimSide * cover * silhouette;

  // 5. Quotient-image multiply: native-resolution texture detail is preserved.
  let calibrated = u.ccm * orig;
  var relit = calibrated * ratio * (1.0 - mc.a * (1.0 - shadow) * u.shadowStrength)
            + spec * shadow * u.specStrength * u.keyColor * mask.g;
  // Filmic shoulder on the *added* light only: values above the knee roll off smoothly towards 1
  // instead of clipping (a clipped patch has a hard outline — the "harsh" nose / jaw highlight).
  if (u.highlightSoft > 0.0) {
    let knee = mix(1.0, 0.6, u.highlightSoft);
    let over = max(relit - vec3f(knee), vec3f(0.0));
    let rolled = vec3f(knee) + (1.0 - knee) * (vec3f(1.0) - exp(-over / (1.0 - knee + 1e-4)));
    relit = max(select(relit, rolled, relit > vec3f(knee)), min(relit, calibrated));
  }

  // Rim is added outside the edge fade (it lives exactly at grazing angles).
  var outLin = mix(orig, relit, alpha) + fresnel * u.rimColor * u.strength;

  // Virtual background: cover-fit image behind the person matte (screen uv → not mirrored), with a
  // little light wrap so the background's light spills over the subject's edge instead of a cut-out.
  if (u.bg.x > 0.5) {
    let ca = gdim.x / gdim.y;
    var buv = in.uv;
    if (u.bg.y > ca) {
      buv.x = 0.5 + (in.uv.x - 0.5) * ca / u.bg.y;
    } else {
      buv.y = 0.5 + (in.uv.y - 0.5) * u.bg.y / ca;
    }
    let bgLin = textureSampleLevel(bgTex, samp, buv, 0.0).rgb;
    // Matte: guided-filter refined (image-edge aligned, temporally stable) when available, else the
    // 256² upscaled person mask choked slightly inward.
    let refined = textureSampleLevel(matteTex, samp, uv, 0.0).r;
    let matte = select(smoothstep(0.45, 0.85, mask.r), smoothstep(0.15, 0.85, refined), u.bg.w > 0.5);
    // Edge pixels are a mix of you and the *camera* background (e.g. a white wall) — that mix is the
    // bright gap. Estimate the camera background just outside the silhouette (along −∇mask) and unmix:
    // F = (I − (1−a)·B) / a.
    var subject = outLin;
    if (matte < 0.995 && matte > 0.0) {
      let mt = 1.0 / vec2f(textureDimensions(maskTex));
      let g = vec2f(
        textureSampleLevel(maskTex, samp, uv + vec2f(mt.x, 0.0), 0.0).r - textureSampleLevel(maskTex, samp, uv - vec2f(mt.x, 0.0), 0.0).r,
        textureSampleLevel(maskTex, samp, uv + vec2f(0.0, mt.y), 0.0).r - textureSampleLevel(maskTex, samp, uv - vec2f(0.0, mt.y), 0.0).r);
      let gl = length(g);
      if (gl > 1e-4) {
        let outward = -g / gl;
        let camBg = toLinear(srcAt(uv + outward * mt * 3.0));
        let a = max(matte, 0.2);
        subject = clamp((outLin - (1.0 - a) * camBg) / a, vec3f(0.0), vec3f(1.5));
      }
    }
    let wrap = clamp(matte * (1.0 - matte) * 4.0, 0.0, 1.0) * u.bg.z;
    subject = mix(subject, subject * 0.6 + bgLin * 0.6, wrap);
    outLin = mix(bgLin, subject, matte);
  }

  let m = u32(u.mode + 0.5);
  if (m == 1u) {
    outLin = orig;
  } else if (m == 2u) {
    outLin = select(outLin, orig, in.uv.x < u.splitX);
  } else if (m == 3u) {
    // Normals weighted by relight coverage, over the dimmed video.
    return vec4f(mix(origSrgb * 0.25, N * 0.5 + 0.5, cover), 1.0);
  } else if (m == 4u) {
    // log2 quotient, diverging: blue = darken (de-light), orange = brighten (add-light).
    let t = clamp(log2(max(dot(ratio, vec3f(0.2126, 0.7152, 0.0722)), 1e-3)) / 2.0, -1.0, 1.0);
    let viz = select(mix(vec3f(0.5), vec3f(0.15, 0.35, 1.0), -t), mix(vec3f(0.5), vec3f(1.0, 0.55, 0.1), t), t > 0.0);
    return vec4f(mix(origSrgb * 0.3, viz, alpha), 1.0);
  } else if (m == 5u) {
    return vec4f(mask.rgb, 1.0);
  } else if (m == 6u) {
    return vec4f(mix(origSrgb * 0.4, vec3f(shadow), select(0.0, 1.0, fuv.b > 0.5)), 1.0);
  }

  var outSrgb = toSrgb(outLin);
  if (m == 2u) {
    // Divider (2 px) + round grab handle at mid-height (drag it on the canvas).
    let px = vec2f(1.0 / max(fwidth(in.uv.x), 1e-6), 1.0 / max(fwidth(in.uv.y), 1e-6));
    let d = (in.uv - vec2f(u.splitX, 0.5)) * px;
    let r = length(d);
    if (abs(d.x) < 1.0) { outSrgb = vec3f(1.0); }
    if (r < 16.0) {
      outSrgb = mix(outSrgb * 0.35, vec3f(1.0), smoothstep(13.0, 14.5, r));
      // little ◀ ▶ chevrons
      let ax = abs(d.x);
      if (ax > 3.0 && ax < 9.0 && abs(d.y) < (9.0 - ax) * 0.8) { outSrgb = vec3f(1.0); }
    }
  }
  return vec4f(clamp(outSrgb, vec3f(0.0), vec3f(1.0)), 1.0);
}
