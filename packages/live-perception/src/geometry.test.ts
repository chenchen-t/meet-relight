import { describe, it, expect } from 'vitest';
import {
  CANONICAL_POSITIONS,
  FACE_VERTEX_COUNT,
  FACE_TRIANGLES,
  buildFaceVertices,
  FACE_VERTEX_STRIDE,
  headPoseFromMatrix,
  shadowLayerWeights,
  OneEuroArray,
  estimateFaceExpandProfile,
  medianProfile,
  EXPAND_BINS,
  expandFaceSilhouette,
} from './index.ts';

/** Orthographically "photograph" the canonical face into MediaPipe-style normalized landmarks. */
function canonicalAsLandmarks(): Float32Array {
  const lm = new Float32Array(FACE_VERTEX_COUNT * 3);
  const S = 24; // cm across the image
  for (let i = 0; i < FACE_VERTEX_COUNT; i++) {
    lm[i * 3] = CANONICAL_POSITIONS[i * 3]! / S + 0.5;
    lm[i * 3 + 1] = 0.5 - CANONICAL_POSITIONS[i * 3 + 1]! / S;
    lm[i * 3 + 2] = -CANONICAL_POSITIONS[i * 3 + 2]! / S; // closer = smaller z
  }
  return lm;
}

describe('face mesh data', () => {
  it('has 468 vertices and 898 triangles with valid indices', () => {
    expect(FACE_VERTEX_COUNT).toBe(468);
    expect(FACE_TRIANGLES.length).toBe(898 * 3);
    expect(Math.max(...FACE_TRIANGLES)).toBeLessThan(468);
  });
});

describe('buildFaceVertices', () => {
  const v = buildFaceVertices(canonicalAsLandmarks(), 1, 1);
  const normal = (i: number) => [v[i * FACE_VERTEX_STRIDE + 3]!, v[i * FACE_VERTEX_STRIDE + 4]!, v[i * FACE_VERTEX_STRIDE + 5]!];

  it('produces outward (camera-facing) normals for a frontal face', () => {
    let meanZ = 0;
    for (let i = 0; i < FACE_VERTEX_COUNT; i++) meanZ += normal(i)[2]!;
    expect(meanZ / FACE_VERTEX_COUNT).toBeGreaterThan(0.5);
  });

  it('nose tip faces the camera; cheeks face sideways in the right direction', () => {
    expect(normal(1)[2]).toBeGreaterThan(0.8); // landmark 1 = nose tip
    // Landmark 234 is on the face's right side (viewer's left in an unmirrored image), 454 opposite.
    expect(normal(234)[0]).toBeLessThan(-0.3);
    expect(normal(454)[0]).toBeGreaterThan(0.3);
  });

  it('forehead normals tilt up, chin normals tilt down', () => {
    expect(normal(10)[1]).toBeGreaterThan(0.2); // forehead top
    expect(normal(152)[1]).toBeLessThan(-0.2); // chin
  });
});

describe('head pose', () => {
  it('reads yaw from the transform matrix', () => {
    const a = (30 * Math.PI) / 180;
    // Rotation about +y by 30°: forward (0,0,1) → (sin a, 0, cos a).
    const m = new Float32Array([Math.cos(a), 0, -Math.sin(a), 0, 0, 1, 0, 0, Math.sin(a), 0, Math.cos(a), 0, 0, 0, -50, 1]);
    const p = headPoseFromMatrix(m);
    expect(p.yawDeg).toBeCloseTo(30, 3);
    expect(p.pitchDeg).toBeCloseTo(0, 3);
  });

  it('maps yaw to shadow layer weights summing to 1', () => {
    for (const y of [-45, -15, 0, 10, 30, 60]) {
      const w = shadowLayerWeights(y);
      expect(w[0] + w[1] + w[2]).toBeCloseTo(1);
    }
    expect(shadowLayerWeights(0)).toEqual([0, 1, 0]);
  });
});

describe('OneEuroArray', () => {
  it('suppresses jitter at rest but tracks large motion', () => {
    const f = new OneEuroArray();
    let t = 0;
    let out = f.filter(new Float32Array([0]), t);
    for (let i = 0; i < 60; i++) {
      t += 1 / 30;
      out = f.filter(new Float32Array([(i % 2 ? 1 : -1) * 0.002]), t);
    }
    expect(Math.abs(out[0]!)).toBeLessThan(0.0015);
    for (let i = 0; i < 10; i++) {
      t += 1 / 30;
      out = f.filter(new Float32Array([0.3]), t);
    }
    expect(out[0]).toBeGreaterThan(0.25);
  });
});

describe('headPoseFromLandmarks', () => {
  it('is ~0 for the frontal canonical face and follows a synthetic yaw', async () => {
    const { headPoseFromLandmarks } = await import('./index.ts');
    const lm = canonicalAsLandmarks();
    const p0 = headPoseFromLandmarks(lm, 1, 1);
    expect(Math.abs(p0.yawDeg)).toBeLessThan(5);
    // Rotate canonical face by +25° about y (towards camera +x) and re-project.
    const a = (25 * Math.PI) / 180;
    const S = 24;
    const rot = new Float32Array(lm.length);
    for (let i = 0; i < FACE_VERTEX_COUNT; i++) {
      const x = CANONICAL_POSITIONS[i * 3]!, y = CANONICAL_POSITIONS[i * 3 + 1]!, z = CANONICAL_POSITIONS[i * 3 + 2]!;
      const xr = Math.cos(a) * x + Math.sin(a) * z;
      const zr = -Math.sin(a) * x + Math.cos(a) * z;
      rot[i * 3] = xr / S + 0.5;
      rot[i * 3 + 1] = 0.5 - y / S;
      rot[i * 3 + 2] = -zr / S;
    }
    const p = headPoseFromLandmarks(rot, 1, 1);
    expect(p.yawDeg - p0.yawDeg).toBeGreaterThan(20);
    expect(p.yawDeg - p0.yawDeg).toBeLessThan(30);
  });
});

describe('headProxyFromLandmarks', () => {
  it('centres above the face, extends past the forehead landmark, and reads roll', async () => {
    const { headProxyFromLandmarks } = await import('./index.ts');
    const lm = canonicalAsLandmarks();
    const h = headProxyFromLandmarks(lm, 1);
    const topY = lm[10 * 3 + 1]!;
    expect(Math.abs(h.roll)).toBeLessThan(0.05);
    expect(h.cy - h.ry).toBeLessThan(topY - 0.02); // proxy reaches above landmark 10 (hairline side)
    expect(h.rx).toBeGreaterThan(0);
    // Roll the image CCW by 20°: rotate landmarks about the image centre (y-down coords → CW in array space).
    const a = (20 * Math.PI) / 180;
    const rl = Float32Array.from(lm);
    for (let i = 0; i < FACE_VERTEX_COUNT; i++) {
      const x = lm[i * 3]! - 0.5, y = -(lm[i * 3 + 1]! - 0.5);
      rl[i * 3] = Math.cos(a) * x - Math.sin(a) * y + 0.5;
      rl[i * 3 + 1] = -(Math.sin(a) * x + Math.cos(a) * y) + 0.5;
    }
    expect(headProxyFromLandmarks(rl, 1).roll).toBeCloseTo(a, 1);
  });
});

describe('OneEuroArray.predict', () => {
  it('extrapolates a constant-velocity signal to cancel latency', () => {
    const f = new OneEuroArray();
    const v = 0.5; // units / s
    let t = 0;
    for (let i = 0; i < 30; i++, t += 1 / 30) f.filter(new Float32Array([v * t]), t);
    const tLast = t - 1 / 30;
    const lag = v * tLast - f.predict(0)![0]!; // smoothing lag
    const err = Math.abs(v * (tLast + 0.05) - f.predict(0.05)![0]!);
    expect(err).toBeLessThan(Math.abs(lag) + 0.002);
    expect(err).toBeLessThan(0.01);
  });
});

describe('OneEuroArray rigidSnap (affine)', () => {
  const move = (lm: Float32Array, dx: number, s: number, rollRad: number) => {
    const o = Float32Array.from(lm);
    for (let i = 0; i < FACE_VERTEX_COUNT; i++) {
      const x = lm[i * 3]! - 0.5, y = lm[i * 3 + 1]! - 0.5;
      o[i * 3] = s * (Math.cos(rollRad) * x - Math.sin(rollRad) * y) + 0.5 + dx;
      o[i * 3 + 1] = s * (Math.sin(rollRad) * x + Math.cos(rollRad) * y) + 0.5;
    }
    return o;
  };
  it('follows translation / scale / roll on the very first moving frame', () => {
    const lm = canonicalAsLandmarks();
    const f = new OneEuroArray();
    let t = 0;
    for (let i = 0; i < 10; i++, t += 1 / 30) f.filter(lm, t);
    const raw = move(lm, 0.08, 1.1, 0.15);
    const out = f.filter(raw, t);
    let maxErr = 0;
    for (let i = 0; i < FACE_VERTEX_COUNT * 3; i += 3) maxErr = Math.max(maxErr, Math.abs(out[i]! - raw[i]!), Math.abs(out[i + 1]! - raw[i + 1]!));
    expect(maxErr).toBeLessThan(5e-4); // vs 0.08 motion; residual = identity regulariser bias
    f.rigidSnap = false;
    const g = new OneEuroArray();
    g.rigidSnap = false;
    t = 0;
    for (let i = 0; i < 10; i++, t += 1 / 30) g.filter(lm, t);
    const lag = g.filter(raw, t);
    expect(Math.abs(lag[0]! - raw[0]!)).toBeGreaterThan(0.01); // plain filter trails
  });
  it('cuts most of the lag of a head turn (yaw is only approximately affine)', () => {
    const lm = canonicalAsLandmarks();
    const S = 24;
    const yaw = (deg: number) => {
      const a = (deg * Math.PI) / 180;
      const o = new Float32Array(lm.length);
      for (let i = 0; i < FACE_VERTEX_COUNT; i++) {
        const x = CANONICAL_POSITIONS[i * 3]!, y = CANONICAL_POSITIONS[i * 3 + 1]!, z = CANONICAL_POSITIONS[i * 3 + 2]!;
        o[i * 3] = (Math.cos(a) * x + Math.sin(a) * z) / S + 0.5;
        o[i * 3 + 1] = 0.5 - y / S;
        o[i * 3 + 2] = -(-Math.sin(a) * x + Math.cos(a) * z) / S;
      }
      return o;
    };
    const err = (snap: boolean) => {
      const f = new OneEuroArray();
      f.rigidSnap = snap;
      let t = 0;
      for (let i = 0; i < 10; i++, t += 1 / 30) f.filter(yaw(0), t);
      let e = 0;
      for (let k = 1; k <= 3; k++, t += 1 / 30) { // 3 frames of a 120°/s turn
        const raw = yaw(4 * k);
        const out = f.filter(raw, t);
        e = 0;
        for (let i = 0; i < lm.length; i += 3) e += Math.hypot(out[i]! - raw[i]!, out[i + 1]! - raw[i + 1]!);
      }
      return e;
    };
    expect(err(true)).toBeLessThan(err(false) * 0.35);
  });
  it('still suppresses per-vertex jitter at rest', () => {
    const lm = canonicalAsLandmarks();
    const f = new OneEuroArray();
    let t = 0;
    let seed = 1;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 0.004;
    let out = f.filter(lm, t);
    let rawErr = 0, outErr = 0;
    for (let k = 0; k < 60; k++) {
      t += 1 / 30;
      const noisy = Float32Array.from(lm, (v) => v + rnd());
      out = f.filter(noisy, t);
      if (k > 30) for (let i = 0; i < lm.length; i++) { rawErr += Math.abs(noisy[i]! - lm[i]!); outErr += Math.abs(out[i]! - lm[i]!); }
    }
    expect(outErr).toBeLessThan(rawErr * 0.6);
  });
});

describe('expandFaceSilhouette', () => {
  it('moves the outer ring outward (and back), leaves the centre alone', async () => {

    const lm = canonicalAsLandmarks();
    const out = expandFaceSilhouette(lm, 100, 100, 0.1);
    // nose tip (1) barely moves; face-oval point 234 (cheek edge) moves outward and back
    expect(Math.abs(out[1 * 3]! - lm[1 * 3]!)).toBeLessThan(1e-3);
    const cx = 0.5;
    expect(Math.abs(out[234 * 3]! - cx)).toBeGreaterThan(Math.abs(lm[234 * 3]! - cx) + 0.005);
    expect(out[234 * 3 + 2]!).toBeGreaterThan(lm[234 * 3 + 2]!);
    expect(expandFaceSilhouette(lm, 100, 100, 0)).toEqual(lm.subarray(0, lm.length));
  });
});

describe('estimateFaceExpandProfile (Greenroom auto-fit)', () => {
  // Synthetic face: landmarks on concentric ellipses (outer ring = mesh outline, radius 100 px) in a 640x480 frame.
  const W = 640, H = 480, CX = 320, CY = 240, R = 100;
  const lm = new Float32Array(FACE_VERTEX_COUNT * 3);
  for (let i = 0; i < FACE_VERTEX_COUNT; i++) {
    const a = (i / FACE_VERTEX_COUNT) * 2 * Math.PI * 7;
    const r = R * (0.3 + 0.7 * ((i % 10) / 9));
    lm[i * 3] = (CX + Math.cos(a) * r) / W;
    lm[i * 3 + 1] = (CY + Math.sin(a) * r) / H;
  }
  // Eye corners level (no roll).
  lm[33 * 3] = (CX - 40) / W; lm[33 * 3 + 1] = (CY - 20) / H;
  lm[263 * 3] = (CX + 40) / W; lm[263 * 3 + 1] = (CY - 20) / H;
  const MW = 256, MH = 256;
  /** Face-skin mask: circle of radius R·sides, extended to R·top above the centre. */
  const mask = (top: number, sides: number) => {
    const m = new Uint8Array(MW * MH * 4);
    for (let v = 0; v < MH; v++)
      for (let u = 0; u < MW; u++) {
        const x = ((u + 0.5) / MW) * W - CX, y = ((v + 0.5) / MH) * H - CY;
        const rr = y < 0 ? R * (sides + (top - sides) * (-y / Math.hypot(x, y) || 0)) : R * sides;
        if (Math.hypot(x, y) <= rr) m[(v * MW + u) * 4 + 2] = 255;
      }
    return m;
  };
  const topBin = Math.floor(0.75 * EXPAND_BINS); // face-aligned angle 270° = straight up (image y down)
  const sideBin = Math.floor(0.125 * EXPAND_BINS); // 45° = lower cheek / jaw

  it('expands upward when skin continues above the mesh (short hair / forehead)', () => {
    const p = estimateFaceExpandProfile(mask(1.14, 1.0), MW, MH, lm, W, H)!;
    expect(p).not.toBeNull();
    expect(p[topBin]!).toBeGreaterThan(0.09);
    expect(Math.abs(p[sideBin]!)).toBeLessThan(0.04);
  });

  it('stays tight when hair covers the forehead (skin ends at the mesh)', () => {
    const p = estimateFaceExpandProfile(mask(1.0, 1.0), MW, MH, lm, W, H)!;
    for (const v of p) expect(Math.abs(v)).toBeLessThan(0.04);
  });

  it('medianProfile rejects a single outlier frame', () => {
    const a = new Float32Array(EXPAND_BINS).fill(0.05), b = new Float32Array(EXPAND_BINS).fill(0.18);
    expect(medianProfile([a, a, b])![0]).toBeCloseTo(0.05);
  });

  it('expandFaceSilhouette follows the profile per direction', () => {
    const prof = new Float32Array(EXPAND_BINS);
    prof[topBin] = prof[topBin - 1] = prof[topBin + 1] = 0.15;
    const out = expandFaceSilhouette(lm, W, H, 0, undefined, prof);
    // Find the outermost landmark pointing up and one pointing right; only the top should move.
    let up = -1, right = -1, bu = 9, br = 9;
    for (let i = 0; i < FACE_VERTEX_COUNT; i++) {
      const x = lm[i * 3]! * W - CX, y = lm[i * 3 + 1]! * H - CY;
      if (Math.hypot(x, y) < R * 0.99) continue;
      const a = Math.atan2(y, x);
      if (Math.abs(a + Math.PI / 2) < bu) { bu = Math.abs(a + Math.PI / 2); up = i; }
      if (Math.abs(a) < br) { br = Math.abs(a); right = i; }
    }
    expect(up).toBeGreaterThan(-1);
    expect((lm[up * 3 + 1]! - out[up * 3 + 1]!) * H).toBeGreaterThan(10);
    if (right >= 0) expect(Math.abs(out[right * 3]! - lm[right * 3]!) * W).toBeLessThan(1);
  });
});
