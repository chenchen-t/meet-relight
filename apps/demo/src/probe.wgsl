struct U {
  viewport: vec2f,   // canvas size in px
  mirror: f32,
  showMask: f32,
  pointSize: f32,
  numPoints: f32,
  _pad: vec2f,
};

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var video: texture_external;
@group(0) @binding(2) var maskTex: texture_2d<f32>;
@group(0) @binding(3) var<uniform> u: U;
@group(0) @binding(4) var<storage, read> points: array<f32>; // xyz per landmark

struct VO {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

fn mirrorUv(uv: vec2f) -> vec2f {
  return select(uv, vec2f(1.0 - uv.x, uv.y), u.mirror > 0.5);
}

@vertex
fn vsFull(@builtin(vertex_index) i: u32) -> VO {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VO;
  o.pos = vec4f(p[i], 0.0, 1.0);
  o.uv = vec2f(p[i].x * 0.5 + 0.5, 0.5 - p[i].y * 0.5);
  return o;
}

@fragment
fn fsFull(in: VO) -> @location(0) vec4f {
  let uv = mirrorUv(in.uv);
  var rgb = textureSampleBaseClampToEdge(video, samp, uv).rgb;
  if (u.showMask > 0.5) {
    let m = textureSampleLevel(maskTex, samp, uv, 0.0);
    // fg = blue tint, skin = orange tint, hair = green tint
    rgb = mix(rgb, vec3f(0.2, 0.4, 1.0), 0.25 * m.r * (1.0 - m.g) * (1.0 - m.a));
    rgb = mix(rgb, vec3f(1.0, 0.55, 0.2), 0.35 * m.g);
    rgb = mix(rgb, vec3f(0.2, 1.0, 0.3), 0.35 * m.a);
  }
  return vec4f(rgb, 1.0);
}

@vertex
fn vsPoint(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let p = mirrorUv(vec2f(points[ii * 3u], points[ii * 3u + 1u]));
  let ndc = vec2f(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0);
  var o: VO;
  o.pos = vec4f(ndc + corners[vi] * u.pointSize / u.viewport, 0.0, 1.0);
  o.uv = corners[vi];
  return o;
}

@fragment
fn fsPoint(in: VO) -> @location(0) vec4f {
  return vec4f(0.3, 1.0, 0.9, 1.0);
}
