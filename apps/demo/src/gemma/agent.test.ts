import { describe, it, expect } from 'vitest';
import { executeTool, parseToolCalls, runTurn, stripToolCalls, toRigAz, toolPrompt, type LightingTarget } from './agent.ts';
import type { LightingPreset } from '../ui.ts';

const base = (): LightingPreset => ({
  name: 'current',
  key: { az: -45, el: 30, intensity: 0.85, softness: 0.75, kelvin: 5200, gel: '#ffffff' },
  fill: { az: 50, el: 10, intensity: 0.27, softness: 1, kelvin: 5600, gel: '#ffffff' },
  rim: { az: 160, el: 20, intensity: 0.2, kelvin: 8000, gel: '#ffffff' },
  ambient: 0.15, roughness: 0.5, spec: 0.15,
});

function fakeTarget(mirrored = true) {
  let cur = base();
  const applied: string[] = [];
  const t: LightingTarget = {
    getPreset: () => JSON.parse(JSON.stringify(cur)),
    applyPreset: (p) => { cur = JSON.parse(JSON.stringify(p)); },
    mirrored: () => mirrored,
    presetNames: () => ['Studio', 'Film noir'],
    applyPresetByName: (n) => { applied.push(n); return n === 'Film noir'; },
    backgroundNames: () => ['Beach'],
    setBackground: async (n) => n === 'Beach',
  };
  return { t, get: () => cur, applied };
}

describe('tool-call parsing', () => {
  it('reads tagged and bare calls, ignores incomplete ones', () => {
    const s = 'ok <tool_call>{"name": "adjust_light", "arguments": {"light": "key"}}</tool_call> and ' +
      '{"name": "apply_preset", "arguments": {"name": "Studio"}} then {"name": "set_lighting", "arguments": {"key": {';
    const calls = parseToolCalls(s);
    expect(calls.map((c) => c.name)).toEqual(['adjust_light', 'apply_preset']);
    expect(stripToolCalls('<tool_call>{"name":"x","arguments":{}}</tool_call> Done.')).toBe('Done.');
  });
});

describe('executeTool', () => {
  it('set_lighting converts picture azimuth to the (mirrored) rig and clamps values', async () => {
    const f = fakeTarget(true);
    await executeTool({ name: 'set_lighting', arguments: {
      key: { azimuth: -60, elevation: 120, intensity: 5, softness: 0.3, kelvin: 3000, color: 'magenta' },
      fill: { azimuth: 70, intensity: 0.4, color: '#00e5ff' },
      rim: { azimuth: 160 }, ambient: 0.05,
    } }, f.t);
    const p = f.get();
    expect(p.key.az).toBe(toRigAz(-60, true)); // picture-left → +60 in the mirrored rig
    expect(p.key.el).toBe(90);
    expect(p.key.intensity).toBe(2);
    expect(p.key.gel).toBe('#ff2bd6');
    expect(p.fill.gel).toBe('#00e5ff');
    expect(p.ambient).toBeCloseTo(0.05);
  });
  it('adjust_light: warmer = lower kelvin on all lights; azimuth deltas in picture space', async () => {
    const f = fakeTarget(true);
    await executeTool({ name: 'adjust_light', arguments: { light: 'all', kelvin_delta: -900 } }, f.t);
    expect(f.get().key.kelvin).toBe(4300);
    expect(f.get().rim.kelvin).toBe(7100);
    const before = f.get().key.az;
    await executeTool({ name: 'adjust_light', arguments: { light: 'key', azimuth_delta: -30 } }, f.t);
    expect(f.get().key.az).toBe(before + 30); // picture-left in a mirrored view = +rig azimuth
  });
  it('errors on unknown presets / tools', async () => {
    const f = fakeTarget();
    await expect(executeTool({ name: 'apply_preset', arguments: { name: 'nope' } }, f.t)).rejects.toThrow();
    await expect(executeTool({ name: 'fly', arguments: {} }, f.t)).rejects.toThrow();
  });
});

describe('runTurn', () => {
  it('executes streamed calls once each and reports the text', async () => {
    const f = fakeTarget();
    const reply = '<tool_call>{"name": "apply_preset", "arguments": {"name": "Film noir"}}</tool_call>\nMoody and high-contrast.';
    const calls: string[] = [];
    let text = '';
    const r = await runTurn(async (onPartial) => {
      for (let i = 10; i <= reply.length; i += 10) onPartial(reply.slice(0, i));
      return reply;
    }, f.t, { onCall: (c, res) => calls.push(`${c.name}:${res instanceof Error ? 'err' : 'ok'}`), onText: (t) => (text = t) });
    expect(r.calls).toBe(1);
    expect(calls).toEqual(['apply_preset:ok']);
    expect(f.applied).toEqual(['Film noir']);
    expect(text).toBe('Moody and high-contrast.');
  });
  it('prompt lists presets, backgrounds and the current rig in picture terms', () => {
    const p = toolPrompt(fakeTarget(true).t);
    expect(p).toContain('PRESETS: Studio, Film noir');
    expect(p).toContain('key: az 45'); // rig −45 mirrored → picture +45 (from the right)
  });
});
