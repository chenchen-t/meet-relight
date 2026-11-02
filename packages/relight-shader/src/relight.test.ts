import { describe, it, expect } from 'vitest';
import { EXAMPLE_PROFILE, type TargetProfile } from '@relight/shared-schema';
import {
  shProject,
  shEval,
  fibonacciSphere,
  bakeQuotientMatcap,
  sampleMatcap,
  matcapTexelNormal,
  matcapUv,
  targetShading,
  roomSH,
  roomShadingFn,
  ROOM_PRESETS,
  toHalf,
  normalize,
  type Vec3,
} from './index.ts';

describe('SH9', () => {
  it('reconstructs a Lambertian room light within a few percent', () => {
    const fn = roomShadingFn(ROOM_PRESETS.window_left!);
    const sh = shProject(fn);
    let maxErr = 0;
    for (const n of fibonacciSphere(500)) {
      if (n[2] < 0) continue; // only camera-facing normals matter
      const a = fn(n);
      const b = shEval(sh, n);
      maxErr = Math.max(maxErr, Math.abs(a[0] - b[0]));
    }
    // Order-2 SH of a clamped cosine has known ringing; bounded well under 0.15 of a ~1.0 light.
    expect(maxErr).toBeLessThan(0.15);
  });
});

describe('matcap mapping', () => {
  it('round-trips texel centre normals through matcapUv', () => {
    const size = 64;
    for (const [i, j] of [[32, 32], [10, 40], [50, 12]] as const) {
      const n = matcapTexelNormal(i, j, size);
      const [u, v] = matcapUv(n);
      expect(Math.floor(u * size)).toBe(i);
      expect(Math.floor(v * size)).toBe(j);
    }
  });

  it('puts camera-up normals at the top row (v small)', () => {
    const [, v] = matcapUv([0, 1, 0]);
    expect(v).toBe(0);
  });
});

describe('quotient relighting', () => {
  const profile: TargetProfile = EXAMPLE_PROFILE;

  it('original · Q(N) ≈ albedo · L_target(N) when the room SH is known', () => {
    const room = { kind: 'sh' as const, sh: roomSH(ROOM_PRESETS.overhead!) };
    const data = bakeQuotientMatcap(profile.lighting_rig, room, { maxRatio: 100, minRoom: 1e-3 });
    const albedo: Vec3 = [0.7, 0.5, 0.4];
    let sumErr = 0;
    let count = 0;
    // Front-facing normals (|N.xy| < 0.85) — where faces live and the room is well lit.
    for (const p of fibonacciSphere(2000)) {
      if (p[2] < 0.5) continue;
      const n = normalize(p);
      const roomL = shEval(room.sh, n);
      if (Math.min(...roomL) < 0.1) continue;
      const observed = albedo.map((a, c) => a * roomL[c]!) as Vec3;
      const q = sampleMatcap(data, 64, n);
      const relit = observed.map((o, c) => o * q[c]!);
      const gt = targetShading(profile.lighting_rig, n).map((t, c) => albedo[c]! * t);
      for (let c = 0; c < 3; c++) sumErr += Math.abs(relit[c]! - gt[c]!) / Math.max(gt[c]!, 0.05);
      count += 3;
    }
    const meanRelErr = sumErr / count;
    expect(meanRelErr).toBeLessThan(0.02); // < 2 % mean relative error from 64x64 quantisation
  });

  it('clamps the ratio in dark room regions (noise guardrail)', () => {
    const room = { kind: 'sh' as const, sh: roomSH(ROOM_PRESETS.overhead!) };
    const data = bakeQuotientMatcap(profile.lighting_rig, room, { maxRatio: 4 });
    let max = 0;
    for (let i = 0; i < data.length; i += 4) max = Math.max(max, data[i]!, data[i + 1]!, data[i + 2]!);
    expect(max).toBeLessThanOrEqual(4);
  });

  it('a colour-skewed room fit cannot cast a strong tint (green-face bug)', () => {
    // Room that is strongly magenta (green channel 3x weaker) — e.g. a bad heartbeat refit.
    const base = roomSH(ROOM_PRESETS.overhead!);
    const sh = Float32Array.from(base, (v, i) => (i % 3 === 1 ? v / 3 : v));
    const white = { ...profile.lighting_rig, lights: profile.lighting_rig.lights.map((l) => ({ ...l, color_rgb: [1, 1, 1] as Vec3 })), ambient_rgb: [0.1, 0.1, 0.1] as Vec3 };
    const data = bakeQuotientMatcap(white, { kind: 'sh', sh }, { maxRatio: 4 });
    let worst = 1;
    for (let i = 0; i < data.length; i += 4) {
      const g = data[i + 1]!, rb = Math.max(data[i]!, data[i + 2]!, 1e-6);
      if (g > 0.05) worst = Math.max(worst, g / rb);
    }
    expect(worst).toBeLessThan(1.35); // ≤ ~1.15² from the room-chroma clamp
  });

  it('skin SSS makes red wrap further than blue past the terminator', () => {
    const key = profile.lighting_rig.lights[0]!;
    const rig = { ...profile.lighting_rig, ambient_rgb: [0, 0, 0] as Vec3, lights: [{ ...key, softness: 0, color_rgb: [1, 1, 1] as Vec3, azimuth_deg: -90, elevation_deg: 0 }] };
    const s = targetShading(rig, [0.05, 0, 1]); // just past terminator on far side
    expect(s[0]).toBeGreaterThan(s[2]);
  });
});

describe('half float', () => {
  it('encodes common values exactly', () => {
    expect(toHalf(0)).toBe(0);
    expect(toHalf(1)).toBe(0x3c00);
    expect(toHalf(-2)).toBe(0xc000);
    expect(toHalf(0.5)).toBe(0x3800);
    expect(toHalf(65504)).toBe(0x7bff);
  });
});
