import { describe, it, expect } from 'vitest';
import { roomSH, ROOM_PRESETS, shEval, fibonacciSphere, type Vec3 } from '@relight/relight-shader';
import { fitRoomLight, exposureGain, SAMPLE_STRIDE } from './shFit.ts';

/** Deterministic PRNG. */
function rng(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

/** Simulate Greenroom samples: camera-facing normals, skin albedo with texture noise, plus outliers. */
function simulate(roomName: string, outlierFrac = 0.1, n = 6000): { samples: Float32Array; albedo: Vec3 } {
  const sh = roomSH(ROOM_PRESETS[roomName]!);
  const albedo: Vec3 = [0.62, 0.4, 0.32];
  const rand = rng(42);
  const pts = fibonacciSphere(n * 3).filter((p) => p[2] > 0.25).slice(0, n);
  const out = new Float32Array(pts.length * SAMPLE_STRIDE);
  pts.forEach((p, s) => {
    const L = shEval(sh, p);
    const tex = 0.9 + 0.2 * rand(); // pores / blotchiness
    let rgb = albedo.map((a, c) => a * tex * Math.max(L[c]!, 0));
    if (rand() < outlierFrac) rgb = rand() < 0.5 ? [0.02, 0.015, 0.01] : [0.95, 0.95, 0.95]; // eyes/brows or speculars
    out.set([p[0], p[1], p[2], rgb[0]!, rgb[1]!, rgb[2]!], s * SAMPLE_STRIDE);
  });
  return { samples: out, albedo };
}

describe('fitRoomLight', () => {
  for (const room of ['overhead', 'window_left', 'desk_lamp']) {
    it(`recovers relative room shading for "${room}" despite 10% outliers`, () => {
      const { samples } = simulate(room);
      const fit = fitRoomLight(samples);
      const gt = roomSH(ROOM_PRESETS[room]!);
      // Compare shading *shape* (scale-free): normalise both by their value at N = +z.
      const front: Vec3 = [0, 0, 1];
      const fN = shEval(fit.shRaw, front);
      const gN = shEval(gt, front);
      let err = 0;
      let cnt = 0;
      for (const p of fibonacciSphere(400)) {
        if (p[2] < 0.4) continue;
        const f = shEval(fit.shRaw, p);
        const g = shEval(gt, p);
        const fl = (0.2126 * f[0] + 0.7152 * f[1] + 0.0722 * f[2]) / (0.2126 * fN[0] + 0.7152 * fN[1] + 0.0722 * fN[2]);
        const gl = (0.2126 * g[0] + 0.7152 * g[1] + 0.0722 * g[2]) / (0.2126 * gN[0] + 0.7152 * gN[1] + 0.0722 * gN[2]);
        err += Math.abs(fl - gl) / Math.max(gl, 0.1);
        cnt++;
      }
      expect(err / cnt).toBeLessThan(0.06);
      expect(fit.inlierCount / fit.sampleCount).toBeGreaterThan(0.8);
      expect(fit.coverage).toBeGreaterThan(0.6);
    });
  }

  it('shRoom divides out skin chroma: DC term becomes neutral-ish', () => {
    const { samples } = simulate('flat', 0);
    const fit = fitRoomLight(samples);
    const dc = [fit.shRoom[0]!, fit.shRoom[1]!, fit.shRoom[2]!];
    expect(Math.abs(dc[0]! - dc[1]!) / dc[1]!).toBeLessThan(0.02);
    expect(Math.abs(dc[2]! - dc[1]!) / dc[1]!).toBeLessThan(0.02);
  });

  it('a prior pulls a sparse refit towards previous coefficients', () => {
    const { samples } = simulate('window_left', 0, 300);
    const prior = fitRoomLight(simulate('window_left', 0).samples).shRaw;
    const fit = fitRoomLight(samples, { prior, ridge: [0.5, 0.5] });
    expect(Math.abs(fit.shRaw[3 * 3]! - prior[3 * 3]!)).toBeLessThan(Math.abs(prior[3 * 3]!) * 0.2);
  });
});

describe('exposureGain', () => {
  it('preserves mean luminance', () => {
    const { samples } = simulate('overhead', 0);
    const g = exposureGain(samples, () => 2);
    expect(g).toBeCloseTo(0.5, 5);
  });
});
