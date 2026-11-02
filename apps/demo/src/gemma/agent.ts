// Gemma 4 lighting agent: text / voice / reference image → tool calls that drive the light rig.
// Tool-call parsing follows the LiteRT samples SAM 2 web demo (toolcalls.ts, Apache-2.0): calls are
// <tool_call>{"name": …, "arguments": {…}}</tool_call> (bare JSON objects are accepted too) and are
// executed as soon as each one is complete in the stream.
import type { LightingPreset } from '../ui.ts';

export type Json = null | boolean | number | string | Json[] | { [k: string]: Json };
export interface ToolCall { name: string; arguments: Record<string, Json> }

/** What the agent can read and change in the app. */
export interface LightingTarget {
  getPreset(): LightingPreset; // rig convention
  applyPreset(p: LightingPreset): void; // rig convention
  mirrored(): boolean;
  presetNames(): string[];
  applyPresetByName(name: string): boolean;
  backgroundNames(): string[];
  setBackground(name: string): Promise<boolean>;
}

// ---------------------------------------------------------------- conventions
// The model works in *picture* terms: azimuth 0 = from the camera, +90 = from the right side of the
// picture, −90 = from the left, ±180 = from behind. The rig azimuth is mirrored in selfie view.
export const toRigAz = (screenAz: number, mirrored: boolean) => (mirrored ? -screenAz : screenAz);
export const toScreenAz = toRigAz; // its own inverse

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const num = (v: Json | undefined, d: number) => (typeof v === 'number' && Number.isFinite(v) ? v : typeof v === 'string' && v.trim() !== '' && Number.isFinite(+v) ? +v : d);
const HEX = /^#?[0-9a-f]{6}$/i;
const NAMED: Record<string, string> = {
  white: '#ffffff', warm: '#ffd2a0', orange: '#ff9a3c', amber: '#ffb347', red: '#ff3b3b', pink: '#ff71ce', magenta: '#ff2bd6',
  purple: '#b967ff', violet: '#8f5bff', blue: '#3a8cff', cyan: '#2ad4ff', teal: '#1fd1b5', green: '#5cff7a', yellow: '#ffe45c', gold: '#ffc857',
};
function color(v: Json | undefined, d: string): string {
  if (typeof v !== 'string') return d;
  const s = v.trim().toLowerCase();
  if (HEX.test(s)) return s.startsWith('#') ? s : `#${s}`;
  return NAMED[s] ?? d;
}
const wrapAz = (a: number) => ((((a + 180) % 360) + 360) % 360) - 180;

type Role = 'key' | 'fill' | 'rim';
const ROLES: Role[] = ['key', 'fill', 'rim'];

// ---------------------------------------------------------------- state line for the prompt
export function describeRig(t: LightingTarget): string {
  const p = t.getPreset();
  const m = t.mirrored();
  const l = (r: Role) => {
    const x = p[r];
    return `${r}: az ${Math.round(toScreenAz(x.az, m))} el ${Math.round(x.el)} int ${x.intensity.toFixed(2)}${x.softness !== undefined ? ` soft ${x.softness.toFixed(2)}` : ''} ${x.kelvin ?? 5600}K ${x.gel ?? '#ffffff'}`;
  };
  return `${l('key')}; ${l('fill')}; ${l('rim')}; ambient ${p.ambient.toFixed(2)}`;
}

// ---------------------------------------------------------------- tools
export const TOOL_DOCS = `Tools (call with <tool_call>{"name": ..., "arguments": {...}}</tool_call>):
- set_lighting: replace the whole rig. arguments: {"key": L, "fill": L, "rim": L, "ambient": 0..0.4, "shine": 0..1}
  where L = {"azimuth": -180..180, "elevation": -60..90, "intensity": 0..2, "softness": 0..1, "kelvin": 2500..9000, "color": "#rrggbb"}
  azimuth: 0 = from the camera, +90 = from the RIGHT of the picture, -90 = from the LEFT, 180 = from behind.
  elevation: 0 = eye level, 45 = high, 90 = overhead, negative = from below. softness 1 = big soft source, 0 = hard.
  key = main light (keep it in FRONT: azimuth between -110 and 110); fill = weaker light on the shadow side (more fill = softer, less shadow);
  rim = back/edge light (azimuth beyond ±120).
- adjust_light: change part of the rig. arguments: {"light": "key"|"fill"|"rim"|"all", "azimuth_delta": deg, "elevation_delta": deg,
  "intensity_scale": factor, "kelvin_delta": K (negative = warmer), "softness_delta": -1..1, "color": "#rrggbb"}
  or {"shadow": "less"|"more"} to soften / deepen the shadows.
- apply_preset: arguments {"name": one of PRESETS}
- set_background: arguments {"name": one of BACKGROUNDS}`;

const EXAMPLES = `Examples:
User: make it warmer
<tool_call>{"name": "adjust_light", "arguments": {"light": "all", "kelvin_delta": -900}}</tool_call>
User: move the main light to the left and a bit higher
<tool_call>{"name": "adjust_light", "arguments": {"light": "key", "azimuth_delta": -40, "elevation_delta": 15}}</tool_call>
User: candle-lit dinner
<tool_call>{"name": "set_lighting", "arguments": {"key": {"azimuth": 40, "elevation": -5, "intensity": 1.1, "softness": 0.7, "kelvin": 2600, "color": "#ffb066"}, "fill": {"azimuth": -60, "elevation": 0, "intensity": 0.08, "softness": 1, "kelvin": 3200, "color": "#ffd2a0"}, "rim": {"azimuth": -150, "elevation": 20, "intensity": 0.3, "softness": 0.5, "kelvin": 2800, "color": "#ffc070"}, "ambient": 0.03, "shine": 0.3}}</tool_call>
User: less shadow
<tool_call>{"name": "adjust_light", "arguments": {"shadow": "less"}}</tool_call>
User: brighter
<tool_call>{"name": "adjust_light", "arguments": {"light": "key", "intensity_scale": 1.25}}</tool_call>`;

export function toolPrompt(t: LightingTarget): string {
  return [
    'You control the virtual studio lights on a person in a live video call (relighting on the GPU).',
    'Turn the user request into tool calls. Prefer adjust_light for small changes and set_lighting for a new look or style',
    '(film, artist, mood, time of day). Colour lights with "color" for stylised looks. After the calls, add one short sentence.',
    TOOL_DOCS,
    `PRESETS: ${t.presetNames().join(', ')}`,
    `BACKGROUNDS: ${t.backgroundNames().join(', ')}`,
    `Current rig: ${describeRig(t)}`,
    EXAMPLES,
  ].join('\n');
}

export function referencePrompt(t: LightingTarget, hints: string): string {
  return [
    'You control the virtual studio lights on a person in a live video call.',
    'The image is a REFERENCE photo. Study how its subject is lit: where the brightest light comes from (left/right/above/below/behind),',
    'how hard the shadows are, the colours of the light, the contrast between lit and shadow side, and any rim/back light.',
    'Reproduce that lighting with exactly one set_lighting call, then one short sentence describing the look.',
    TOOL_DOCS,
    hints ? `Measured from the image (use as a hint): ${hints}` : '',
    `Current rig: ${describeRig(t)}`,
  ].filter(Boolean).join('\n');
}

function lightFrom(v: Json | undefined, base: LightingPreset['key'], mirrored: boolean): LightingPreset['key'] {
  const o = (v && typeof v === 'object' && !Array.isArray(v) ? v : {}) as Record<string, Json>;
  return {
    az: Math.round(toRigAz(clamp(num(o.azimuth, toScreenAz(base.az, mirrored)), -180, 180), mirrored)),
    el: Math.round(clamp(num(o.elevation, base.el), -60, 90)),
    intensity: +clamp(num(o.intensity, base.intensity), 0, 2).toFixed(2),
    softness: +clamp(num(o.softness, base.softness ?? 0.7), 0, 1).toFixed(2),
    kelvin: Math.round(clamp(num(o.kelvin, base.kelvin ?? 5600), 2500, 9000) / 50) * 50,
    gel: color(o.color, base.gel ?? '#ffffff'),
  };
}

/** Execute one tool call against the app; returns a short result for the log. */
export async function executeTool(call: ToolCall, t: LightingTarget): Promise<string> {
  const a = call.arguments;
  const m = t.mirrored();
  switch (call.name) {
    case 'set_lighting': {
      const cur = t.getPreset();
      const p: LightingPreset = {
        name: 'Gemma',
        key: lightFrom(a.key, cur.key, m),
        fill: lightFrom(a.fill, cur.fill, m),
        rim: lightFrom(a.rim, cur.rim, m),
        ambient: +clamp(num(a.ambient, cur.ambient), 0, 0.4).toFixed(3),
        roughness: +clamp(1 - num(a.shine, 1 - (cur.roughness ?? 0.5)), 0.05, 1).toFixed(2),
        spec: +clamp(num(a.shine, cur.spec ?? 0.15) * 0.6, 0, 1).toFixed(2),
      };
      // A "key" behind the subject lights nothing on the face; keep it in front (picture ±110°).
      const keyScreen = toScreenAz(p.key.az, m);
      if (Math.abs(keyScreen) > 110) p.key.az = Math.round(toRigAz(Math.sign(keyScreen || 1) * 110, m));
      t.applyPreset(p);
      return `new rig — key ${Math.round(toScreenAz(p.key.az, m))}°/${p.key.el}° ${p.key.gel}, fill ${p.fill.gel}, rim ${p.rim.gel}`;
    }
    case 'adjust_light': {
      const p = t.getPreset();
      // Semantic shadow control: small models often get the direction of a numeric fill scale wrong.
      const shadow = String(a.shadow ?? '').toLowerCase();
      if (shadow === 'less' || shadow === 'more') {
        const k = shadow === 'less' ? 1.8 : 0.5;
        p.fill.intensity = +clamp(Math.max(p.fill.intensity, 0.05) * k, 0, 2).toFixed(2);
        p.ambient = +clamp(p.ambient + (shadow === 'less' ? 0.04 : -0.04), 0, 0.4).toFixed(3);
        if (a.light === undefined) {
          t.applyPreset(p);
          return `${shadow} shadow (fill ×${k})`;
        }
      }
      const which = String(a.light ?? 'key').toLowerCase();
      const roles = which === 'all' ? ROLES : ROLES.filter((r) => r === which);
      if (!roles.length) throw new Error(`unknown light "${which}"`);
      for (const r of roles) {
        const l = p[r];
        const dAz = num(a.azimuth_delta, 0);
        l.az = Math.round(wrapAz(toRigAz(toScreenAz(l.az, m) + dAz, m)));
        l.el = Math.round(clamp(l.el + num(a.elevation_delta, 0), -60, 90));
        l.intensity = +clamp(l.intensity * num(a.intensity_scale, 1), 0, 2).toFixed(2);
        l.kelvin = Math.round(clamp((l.kelvin ?? 5600) + num(a.kelvin_delta, 0), 2500, 9000) / 50) * 50;
        if (l.softness !== undefined) l.softness = +clamp(l.softness + num(a.softness_delta, 0), 0, 1).toFixed(2);
        if (a.color !== undefined) l.gel = color(a.color, l.gel ?? '#ffffff');
      }
      t.applyPreset(p);
      return `adjusted ${roles.join('+')}`;
    }
    case 'apply_preset': {
      const name = String(a.name ?? '');
      if (!t.applyPresetByName(name)) throw new Error(`no preset "${name}"`);
      return `preset ${name}`;
    }
    case 'set_background': {
      const name = String(a.name ?? '');
      if (!(await t.setBackground(name))) throw new Error(`no background "${name}"`);
      return `background ${name}`;
    }
    default:
      throw new Error(`unknown tool ${call.name}`);
  }
}

// ---------------------------------------------------------------- parsing (streaming-safe)
const CALL_RE = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;
const BARE_RE = /\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"(?:arguments|parameters)"\s*:\s*\{/g;

function readObject(text: string, start: number): string | null {
  let depth = 0, inStr = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (ch === '\\') i++;
      else if (ch === '"') inStr = false;
    } else if (ch === '"') inStr = true;
    else if (ch === '{') depth++;
    else if (ch === '}' && --depth === 0) return text.slice(start, i + 1);
  }
  return null;
}

function asCall(json: string): ToolCall | null {
  try {
    const o = JSON.parse(json) as { name?: unknown; arguments?: unknown; parameters?: unknown };
    if (typeof o?.name !== 'string') return null;
    const args = (o.arguments ?? o.parameters ?? {}) as Record<string, Json>;
    return { name: o.name, arguments: args && typeof args === 'object' && !Array.isArray(args) ? args : {} };
  } catch {
    return null;
  }
}

/** Complete tool calls in a (possibly partial) reply, in order. */
export function parseToolCalls(text: string): ToolCall[] {
  const found: Array<{ at: number; call: ToolCall }> = [];
  const taken: Array<[number, number]> = [];
  for (const mm of text.matchAll(CALL_RE)) {
    const inner = mm[1]!.replace(/^```\w*\s*|\s*```$/g, '');
    const b = inner.indexOf('{');
    const obj = b >= 0 ? readObject(inner, b) : null;
    const call = obj ? asCall(obj) : null;
    if (call) {
      found.push({ at: mm.index!, call });
      taken.push([mm.index!, mm.index! + mm[0].length]);
    }
  }
  for (const mm of text.matchAll(BARE_RE)) {
    const at = mm.index!;
    if (taken.some(([x, y]) => at >= x && at < y)) continue;
    const obj = readObject(text, at);
    const call = obj ? asCall(obj) : null;
    if (call) found.push({ at, call });
  }
  return found.sort((x, y) => x.at - y.at).map((f) => f.call);
}

export function stripToolCalls(text: string): string {
  let s = text.replace(CALL_RE, '');
  for (const mm of [...s.matchAll(BARE_RE)].reverse()) {
    const obj = readObject(s, mm.index!);
    if (obj) s = s.slice(0, mm.index!) + s.slice(mm.index! + obj.length);
  }
  return s.replace(/```\w*\s*```/g, '').replace(/<\/?tool_call>/g, '').trim();
}

// ---------------------------------------------------------------- one agent turn
export interface TurnLog { onCall(call: ToolCall, result: string | Error): void; onText(text: string): void }

/**
 * Runs one turn: `generate` streams the model's reply; each complete tool call is executed as soon as it
 * appears (duplicates ignored), then the remaining text is reported.
 */
export async function runTurn(
  generate: (onPartial: (raw: string) => void) => Promise<string>,
  t: LightingTarget,
  log: TurnLog,
  maxCalls = 4,
): Promise<{ calls: number; raw: string }> {
  const done: string[] = [];
  let chain = Promise.resolve();
  const take = (raw: string) => {
    for (const c of parseToolCalls(raw)) {
      const sig = JSON.stringify(c);
      if (done.includes(sig) || done.length >= maxCalls) continue;
      done.push(sig);
      chain = chain.then(async () => {
        try { log.onCall(c, await executeTool(c, t)); } catch (e) { log.onCall(c, e as Error); }
      });
    }
  };
  const raw = await generate(take);
  take(raw);
  await chain;
  const text = stripToolCalls(raw);
  if (text) log.onText(text);
  return { calls: done.length, raw };
}
