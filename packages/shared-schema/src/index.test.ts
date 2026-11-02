import { describe, it, expect } from 'vitest';
import {
  EXAMPLE_PROFILE,
  validateTargetProfile,
  lightDirection,
  azimuthToClock,
} from './index.ts';

describe('TargetProfile schema', () => {
  it('accepts the design-doc example profile', () => {
    const r = validateTargetProfile(EXAMPLE_PROFILE);
    expect(r.errors).toEqual([]);
    expect(r.ok).toBe(true);
  });

  it('rejects more than 3 lights and bad enums', () => {
    const bad = structuredClone(EXAMPLE_PROFILE) as any;
    bad.lighting_rig.lights = [0, 1, 2, 3].map(() => ({ ...EXAMPLE_PROFILE.lighting_rig.lights[0] }));
    bad.lighting_rig.lights[0].role = 'hair';
    const r = validateTargetProfile(bad);
    expect(r.ok).toBe(false);
    expect(r.errors.length).toBeGreaterThanOrEqual(2);
  });
});

describe('light conventions', () => {
  it('maps azimuth/elevation to camera-space unit vectors', () => {
    const front = lightDirection(0, 0);
    expect(front[2]).toBeCloseTo(1);
    const left = lightDirection(-90, 0); // subject's right = viewer's left
    expect(left[0]).toBeCloseTo(-1);
    const top = lightDirection(0, 90);
    expect(top[1]).toBeCloseTo(1);
  });

  it('maps azimuth to clock directions', () => {
    expect(azimuthToClock(0)).toBe('12:00');
    expect(azimuthToClock(90)).toBe('3:00');
    expect(azimuthToClock(-90)).toBe('9:00');
    expect(azimuthToClock(-45)).toBe('10:30');
    expect(azimuthToClock(180)).toBe('6:00');
  });
});
