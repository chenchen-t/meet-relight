import { shBasis, type SH9 } from '@relight/relight-shader';
import { SampleGatherer } from './gatherer.ts';
import { fitRoomLight, SAMPLE_STRIDE, type RoomFit } from './shFit.ts';

export type GreenroomState = 'idle' | 'capturing' | 'ready';

export interface GreenroomOptions {
  captureMs: number;
  heartbeatMs: number;
  monitorMs: number;
  /** Relative mean-luminance change that triggers an immediate refit. */
  jumpThreshold: number;
  crossfadeMs: number;
  /** Heartbeat refits smaller than this (scale-free shading change) are ignored. */
  minShapeChange: number;
  maxSamples: number;
}

export const DEFAULT_GREENROOM: GreenroomOptions = {
  captureMs: 1500,
  heartbeatMs: 5000,
  monitorMs: 500,
  jumpThreshold: 0.15,
  crossfadeMs: 1000,
  minShapeChange: 0.04,
  maxSamples: 60000,
};

/**
 * Design doc §2.3:
 *   Stage A — Greenroom capture (~1.5 s of head turning) → robust SH9 room fit.
 *   Stage C — 0.2 fps heartbeat: re-gather every 5 s (or on a >15 % luminance
 *             jump, e.g. a white slide), refit with the previous fit as prior,
 *             and cross-fade SH coefficients over 1 s.
 */
export class GreenroomController {
  state: GreenroomState = 'idle';
  fit: RoomFit | null = null;
  /** Most recent samples (used for exposure normalisation). */
  samples: Float32Array = new Float32Array(0);
  progress = 0;
  lastEvent = '';
  refits = 0;
  onChange: () => void = () => {};

  private captureStart = 0;
  private captured: Float32Array[] = [];
  private lastMonitor = 0;
  private lastRefit = 0;
  private fadeFrom: SH9 | null = null;
  private fadeTo: SH9 | null = null;
  private fadeStart = 0;
  private gatherer: SampleGatherer;

  constructor(
    device: GPUDevice,
    kind: 'external' | 'texture',
    public opts: GreenroomOptions = { ...DEFAULT_GREENROOM },
  ) {
    this.gatherer = new SampleGatherer(device, kind);
  }

  start(): void {
    this.state = 'capturing';
    this.captureStart = performance.now();
    this.captured = [];
    this.progress = 0;
    this.lastEvent = 'capturing…';
  }

  reset(): void {
    this.state = 'idle';
    this.fit = null;
    this.fadeFrom = this.fadeTo = null;
    this.lastEvent = '';
  }

  /** Current room SH (cross-faded), or null if not fitted. */
  roomSH(now = performance.now()): SH9 | null {
    if (!this.fadeTo) return null;
    if (!this.fadeFrom) return this.fadeTo;
    const t = Math.min(1, (now - this.fadeStart) / this.opts.crossfadeMs);
    if (t >= 1) {
      this.fadeFrom = null;
      return this.fadeTo;
    }
    const s = t * t * (3 - 2 * t);
    const out = new Float32Array(27);
    for (let i = 0; i < 27; i++) out[i] = this.fadeFrom[i]! * (1 - s) + this.fadeTo[i]! * s;
    return out;
  }

  get fading(): boolean {
    return !!this.fadeFrom;
  }

  /**
   * Call once per frame where fresh geometry for `source` is available.
   * Records a gather pass if one is due; returns a post-submit callback.
   */
  tick(
    encoder: GPUCommandEncoder,
    source: GPUExternalTexture | GPUTextureView,
    geo: GPUTexture,
    mask: GPUTextureView,
    hasFace: boolean,
  ): (() => void) | null {
    const now = performance.now();
    if (!hasFace || this.gatherer.pending > 1) return null;
    if (this.state === 'capturing') {
      this.progress = Math.min(1, (now - this.captureStart) / this.opts.captureMs);
      const read = this.gatherer.encode(encoder, source, geo, mask);
      const done = this.progress >= 1;
      return () => {
        read().then((s) => {
          this.captured.push(s);
          if (done && this.state === 'capturing') this.finishCapture();
        });
      };
    }
    if (this.state === 'ready' && now - this.lastMonitor > this.opts.monitorMs) {
      this.lastMonitor = now;
      const read = this.gatherer.encode(encoder, source, geo, mask);
      return () => {
        read().then((s) => this.monitor(s));
      };
    }
    return null;
  }

  private finishCapture(): void {
    let total = 0;
    for (const c of this.captured) total += c.length;
    let all = new Float32Array(total);
    let o = 0;
    for (const c of this.captured) {
      all.set(c, o);
      o += c.length;
    }
    // Subsample uniformly if huge.
    const n = all.length / SAMPLE_STRIDE;
    if (n > this.opts.maxSamples) {
      const step = n / this.opts.maxSamples;
      const sub = new Float32Array(this.opts.maxSamples * SAMPLE_STRIDE);
      for (let i = 0; i < this.opts.maxSamples; i++) sub.set(all.subarray(Math.floor(i * step) * SAMPLE_STRIDE, Math.floor(i * step) * SAMPLE_STRIDE + SAMPLE_STRIDE), i * SAMPLE_STRIDE);
      all = sub;
    }
    try {
      const t0 = performance.now();
      this.fit = fitRoomLight(all);
      this.samples = all;
      this.fadeFrom = null;
      this.fadeTo = this.fit.shRoom;
      this.state = 'ready';
      this.lastRefit = this.lastMonitor = performance.now();
      this.lastEvent = `fitted ${this.captured.length} frames, ${this.fit.inlierCount}/${this.fit.sampleCount} inliers in ${(performance.now() - t0).toFixed(1)} ms`;
    } catch (e) {
      this.state = 'idle';
      this.lastEvent = `fit failed: ${e instanceof Error ? e.message : e}`;
    }
    this.onChange();
  }

  private monitor(s: Float32Array): void {
    if (!this.fit || s.length < SAMPLE_STRIDE * 50) return;
    let lum = 0;
    const n = s.length / SAMPLE_STRIDE;
    for (let i = 0; i < n; i++) {
      const o = i * SAMPLE_STRIDE;
      lum += 0.2126 * s[o + 3]! + 0.7152 * s[o + 4]! + 0.0722 * s[o + 5]!;
    }
    lum /= n;
    const jump = Math.abs(lum - this.fit.meanLum) / Math.max(this.fit.meanLum, 1e-4);
    const now = performance.now();
    const due = now - this.lastRefit > this.opts.heartbeatMs;
    if (!due && jump < this.opts.jumpThreshold) return;
    try {
      // Single-view refit: strongly regularised towards the current fit; DC (overall level) is free.
      const fit = fitRoomLight(s, { prior: this.fit.shRaw, ridge: [0.05, 0.2] });
      if (!fit.shRoom.every(Number.isFinite)) {
        this.lastRefit = now;
        this.lastEvent = 'heartbeat: degenerate fit rejected';
        return;
      }
      const isJump = jump >= this.opts.jumpThreshold;
      const change = shapeChange(this.fadeTo ?? this.fit.shRoom, fit.shRoom);
      this.lastRefit = now;
      if (!isJump && change < this.opts.minShapeChange) {
        this.lastEvent = `heartbeat: stable (Δ ${(change * 100).toFixed(1)}%)`;
        return;
      }
      this.fadeFrom = this.roomSH(now);
      this.fadeTo = fit.shRoom;
      this.fadeStart = now;
      this.fit = { ...fit, coverage: this.fit.coverage };
      this.samples = s;
      this.refits++;
      this.lastEvent = isJump ? `luminance jump ${(jump * 100).toFixed(0)}% → refit` : `heartbeat refit (Δ ${(change * 100).toFixed(1)}%)`;
      this.onChange();
    } catch {
      /* too few samples this beat; skip */
    }
  }
}

const FRONT_NORMALS: [number, number, number][] = (() => {
  const out: [number, number, number][] = [];
  for (let i = -3; i <= 3; i++)
    for (let j = -3; j <= 3; j++) {
      const x = i * 0.12, y = j * 0.12;
      out.push([x, y, Math.sqrt(1 - x * x - y * y)]);
    }
  return out;
})();

/** Scale-free difference in shading shape over camera-facing normals (mean relative). */
export function shapeChange(a: SH9, b: SH9): number {
  const y = new Float32Array(9);
  const lumAt = (sh: SH9, n: [number, number, number]) => {
    shBasis(n, y);
    let l = 0;
    for (let i = 0; i < 9; i++) l += (0.2126 * sh[i * 3]! + 0.7152 * sh[i * 3 + 1]! + 0.0722 * sh[i * 3 + 2]!) * y[i]!;
    return l;
  };
  const a0 = Math.max(lumAt(a, [0, 0, 1]), 1e-6);
  const b0 = Math.max(lumAt(b, [0, 0, 1]), 1e-6);
  let d = 0;
  for (const n of FRONT_NORMALS) {
    const va = lumAt(a, n) / a0;
    d += Math.abs(lumAt(b, n) / b0 - va) / Math.max(va, 0.1);
  }
  return d / FRONT_NORMALS.length;
}
