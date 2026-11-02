import { azimuthToClock, type Light, type TargetProfile } from '@relight/shared-schema';
import { kelvinToRgb, MATCAP_SIZE, linearToSrgb, srgbToLinear } from '@relight/relight-shader';

export type DirtyFlags = Record<string, boolean>;

/** Minimal declarative control-panel kit shared by demo pages. Values can be overridden via URL query (?label=value). */
export function createPanel(panel: HTMLElement, onChange: (what: string[]) => void) {
  const query = new URLSearchParams(location.search);
  const group = (title: string): HTMLFieldSetElement => {
    const fs = document.createElement('fieldset');
    fs.innerHTML = `<legend>${title}</legend>`;
    panel.appendChild(fs);
    return fs;
  };
  const slider = (parent: HTMLElement, label: string, min: number, max: number, step: number, value: number, what: string[] = ['bake', 'params']): Slider => {
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = `<span>${label}</span><input type="range" min="${min}" max="${max}" step="${step}" value="${value}"><output>${value}</output>`;
    const input = row.querySelector('input')!;
    const out = row.querySelector('output')!;
    input.addEventListener('input', () => {
      out.textContent = input.value;
      onChange(what);
    });
    parent.appendChild(row);
    const get = (() => Number(input.value)) as Slider;
    get.set = (v: number) => {
      input.value = String(v);
      out.textContent = input.value;
      onChange(what);
    };
    return get;
  };
  const select = <T extends string>(parent: HTMLElement, label: string, options: Record<T, string>, value: NoInfer<T>, what: string[]): (() => T) => {
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = `<span>${label}</span><select style="grid-column: span 2"></select>`;
    const sel = row.querySelector('select')!;
    const q = query.get(label);
    if (q && q in options) value = q as NoInfer<T>;
    for (const [k, v] of Object.entries(options)) sel.add(new Option(v as string, k, false, k === value));
    sel.addEventListener('change', () => onChange(what));
    parent.appendChild(row);
    return () => sel.value as T;
  };
  const checkbox = (parent: HTMLElement, label: string, value: boolean, what: string[]): Checkbox => {
    const row = document.createElement('label');
    row.style.display = 'block';
    const q = query.get(label.split(' ')[0]!);
    if (q !== null) value = q === '1' || q === 'true';
    row.innerHTML = `<input type="checkbox" ${value ? 'checked' : ''}> ${label}`;
    const input = row.querySelector('input')!;
    input.addEventListener('change', () => onChange(what));
    parent.appendChild(row);
    const get = (() => input.checked) as Checkbox;
    get.set = (v: boolean) => {
      if (input.checked === v) return;
      input.checked = v;
      onChange(what);
    };
    return get;
  };
  /** Colour picker → linear RGB in [0,1]. */
  const color = (parent: HTMLElement, label: string, value: string, what: string[] = ['bake', 'params']): ColorControl => {
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = `<span>${label}</span><input type="color" value="${value}" style="width:100%;height:20px;border:0;background:none;padding:0"><output>${value}</output>`;
    const input = row.querySelector('input')!;
    const out = row.querySelector('output')!;
    input.addEventListener('input', () => {
      out.textContent = input.value;
      onChange(what);
    });
    parent.appendChild(row);
    const get = (() => {
      const h = input.value;
      const srgb = [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
      return srgb.map(srgbToLinear) as [number, number, number];
    }) as ColorControl;
    get.set = (hex: string) => {
      input.value = hex;
      out.textContent = hex;
      onChange(what);
    };
    get.hex = () => input.value;
    return get;
  };
  return { group, slider, select, checkbox, color };
}

export type Panel = ReturnType<typeof createPanel>;

/** Key / fill / rim / ambient controls → schema-valid TargetProfile. */
export function createRigControls(p: Panel, w: string[] = ['bake', 'params']) {
  const gKey = p.group('Key light');
  const key = {
    az: p.slider(gKey, 'azimuth°', -180, 180, 1, -45, w),
    el: p.slider(gKey, 'elevation°', -60, 90, 1, 30, w),
    intensity: p.slider(gKey, 'intensity', 0, 2, 0.01, 0.85, w),
    softness: p.slider(gKey, 'softness', 0, 1, 0.01, 0.75, w),
    kelvin: p.slider(gKey, 'kelvin', 2500, 9000, 50, 5200, w),
    gel: p.color(gKey, 'gel colour', '#ffffff', w),
  };
  const gFill = p.group('Fill light');
  const fill = {
    az: p.slider(gFill, 'azimuth°', -180, 180, 1, 50, w),
    el: p.slider(gFill, 'elevation°', -60, 90, 1, 10, w),
    intensity: p.slider(gFill, 'intensity', 0, 2, 0.01, 0.27, w),
    softness: p.slider(gFill, 'softness', 0, 1, 0.01, 1, w),
    kelvin: p.slider(gFill, 'kelvin', 2500, 12000, 50, 5600, w),
    gel: p.color(gFill, 'gel colour', '#ffffff', w),
  };
  const gRim = p.group('Rim light (Fresnel)');
  const rim = {
    az: p.slider(gRim, 'azimuth°', -180, 180, 1, 160, w),
    el: p.slider(gRim, 'elevation°', -60, 90, 1, 20, w),
    intensity: p.slider(gRim, 'intensity', 0, 2, 0.01, 0.2, w),
    kelvin: p.slider(gRim, 'kelvin', 2500, 12000, 50, 8000, w),
    gel: p.color(gRim, 'gel colour', '#ffffff', w),
  };
  const gMat = p.group('Ambient & material');
  const mat = {
    ambient: p.slider(gMat, 'ambient', 0, 0.6, 0.01, 0.15, w),
    roughness: p.slider(gMat, 'roughness', 0.05, 1, 0.01, 0.5, ['params']),
    spec: p.slider(gMat, 'spec strength', 0, 1, 0.01, 0.15, ['params']),
  };

  /** Light colour = black-body(kelvin) × gel, normalised so the brightest channel is 1. */
  const lightColor = (kelvin: number, gel: [number, number, number]): [number, number, number] => {
    const k = kelvinToRgb(kelvin);
    const c = k.map((v, i) => v * gel[i]!) as [number, number, number];
    const m = Math.max(...c, 1e-4);
    return c.map((v) => +(v / m).toFixed(3)) as [number, number, number];
  };

  const buildProfile = (): TargetProfile => {
    const mk = (id: string, role: Light['role'], az: number, el: number, intensity: number, softness: number, kelvin: number, gel: [number, number, number], shape: Light['shape']): Light => ({
      id, role, azimuth_deg: az, elevation_deg: el, clock_direction: azimuthToClock(az), intensity,
      color_rgb: lightColor(kelvin, gel),
      kelvin, softness, shape, assigned_execution: 'virtual_shader',
    });
    const a = mat.ambient();
    return {
      subject_type: 'portrait',
      lighting_rig: {
        ambient_rgb: [a, a, a * 1.05],
        key_to_fill_ratio: +(key.intensity() / Math.max(fill.intensity(), 1e-3)).toFixed(2),
        lights: [
          mk('key_light', 'key', key.az(), key.el(), key.intensity(), key.softness(), key.kelvin(), key.gel(), 'octagon'),
          mk('fill_light', 'fill', fill.az(), fill.el(), fill.intensity(), fill.softness(), fill.kelvin(), fill.gel(), 'rect_softbox'),
          mk('rim_light', 'rim', rim.az(), rim.el(), rim.intensity(), 0.3, rim.kelvin(), rim.gel(), 'stripbox'),
        ],
      },
      color_dna: { lab_mean: [62.5, 4.2, 8.1], lab_std: [18.4, 6.1, 9.5] },
    };
  };
  const rimColor = (): [number, number, number] =>
    lightColor(rim.kelvin(), rim.gel()).map((c) => c * rim.intensity() * 0.6) as [number, number, number];

  /** Position controls per role, for direct manipulation (light dome). */
  const positions: Record<Light['role'], { az: Slider; el: Slider }> = { key, fill, rim };

  /** Apply a preset; azSign = −1 mirrors it left↔right. */
  const applyPreset = (pr: LightingPreset, azSign = 1) => {
    const setL = (c: { az: Slider; el: Slider; intensity: Slider; kelvin: Slider; gel: ColorControl; softness?: Slider }, l: PresetLight) => {
      c.az.set(l.az * azSign);
      c.el.set(l.el);
      c.intensity.set(l.intensity);
      c.kelvin.set(l.kelvin ?? 5600);
      c.gel.set(l.gel ?? '#ffffff');
      if (c.softness && l.softness !== undefined) c.softness.set(l.softness);
    };
    setL(key, pr.key);
    setL(fill, pr.fill);
    setL(rim, pr.rim);
    mat.ambient.set(pr.ambient);
    mat.roughness.set(pr.roughness ?? 0.5);
    mat.spec.set(pr.spec ?? 0.15);
  };
  /** The current rig as a preset (rig azimuth convention). */
  const getPreset = (): LightingPreset => {
    const getL = (c: { az: Slider; el: Slider; intensity: Slider; kelvin: Slider; gel: ColorControl; softness?: Slider }): PresetLight => ({
      az: c.az(), el: c.el(), intensity: c.intensity(), kelvin: c.kelvin(), gel: c.gel.hex(),
      ...(c.softness ? { softness: c.softness() } : {}),
    });
    return {
      name: 'current', key: getL(key), fill: getL(fill), rim: getL(rim),
      ambient: mat.ambient(), roughness: mat.roughness(), spec: mat.spec(),
    };
  };
  return { buildProfile, rimColor, roughness: mat.roughness, spec: mat.spec, positions, applyPreset, getPreset };
}

export type Slider = (() => number) & { set: (v: number) => void };
export type Checkbox = (() => boolean) & { set: (v: boolean) => void };
export type ColorControl = (() => [number, number, number]) & { set: (hex: string) => void; hex: () => string };

interface PresetLight { az: number; el: number; intensity: number; softness?: number; kelvin?: number; gel?: string }
export interface LightingPreset {
  name: string;
  key: PresetLight;
  fill: PresetLight;
  rim: PresetLight;
  ambient: number;
  roughness?: number;
  spec?: number;
}

/** Azimuth: − = camera-left of subject (before mirroring), elevation: + = above eye line. */
export const LIGHTING_PRESETS: LightingPreset[] = [
  {
    name: '🎬 Studio',
    key: { az: -45, el: 30, intensity: 0.85, softness: 0.75, kelvin: 5200 },
    fill: { az: 50, el: 10, intensity: 0.27, softness: 1, kelvin: 5600 },
    rim: { az: 160, el: 20, intensity: 0.2, kelvin: 8000 },
    ambient: 0.15,
  },
  {
    name: '✨ Soft beauty',
    key: { az: -15, el: 25, intensity: 0.8, softness: 1, kelvin: 5600 },
    fill: { az: 35, el: 0, intensity: 0.55, softness: 1, kelvin: 5600 },
    rim: { az: 170, el: 30, intensity: 0.35, kelvin: 6500 },
    ambient: 0.22, roughness: 0.6, spec: 0.08,
  },
  {
    name: '🖼 Rembrandt',
    key: { az: -50, el: 40, intensity: 1.15, softness: 0.45, kelvin: 4500 },
    fill: { az: 60, el: 5, intensity: 0.1, softness: 1, kelvin: 5000 },
    rim: { az: 150, el: 25, intensity: 0.15, kelvin: 5000 },
    ambient: 0.07, roughness: 0.45, spec: 0.2,
  },
  {
    name: '🦋 Butterfly',
    key: { az: 0, el: 50, intensity: 1.0, softness: 0.6, kelvin: 5600 },
    fill: { az: 0, el: -25, intensity: 0.3, softness: 1, kelvin: 5600 },
    rim: { az: 180, el: 30, intensity: 0.25, kelvin: 7000 },
    ambient: 0.12, spec: 0.2,
  },
  {
    name: '🌗 Split',
    key: { az: -90, el: 10, intensity: 1.25, softness: 0.4, kelvin: 5200 },
    fill: { az: 90, el: 0, intensity: 0.04, softness: 1, kelvin: 5600 },
    rim: { az: 120, el: 15, intensity: 0.1, kelvin: 8000 },
    ambient: 0.04,
  },
  {
    name: '🌆 Cyberpunk',
    key: { az: -70, el: 15, intensity: 1.15, softness: 0.9, kelvin: 6500, gel: '#ff3cc8' },
    fill: { az: 75, el: 5, intensity: 0.6, softness: 0.95, kelvin: 6500, gel: '#3a8cff' },
    rim: { az: 150, el: 20, intensity: 1.3, kelvin: 6500, gel: '#2ad4ff' },
    ambient: 0.05, roughness: 0.35, spec: 0.3,
  },
  {
    name: '🌴 Vaporwave',
    key: { az: -40, el: 25, intensity: 0.9, softness: 0.9, kelvin: 6500, gel: '#ff9ad8' },
    fill: { az: 60, el: 10, intensity: 0.5, softness: 1, kelvin: 6500, gel: '#a77bff' },
    rim: { az: 165, el: 25, intensity: 1.1, kelvin: 6500, gel: '#01cdfe' },
    ambient: 0.1, roughness: 0.5, spec: 0.2,
  },
  {
    name: '🌅 Golden hour',
    key: { az: -65, el: 8, intensity: 1.2, softness: 0.7, kelvin: 3000, gel: '#ffc890' },
    fill: { az: 60, el: 15, intensity: 0.22, softness: 1, kelvin: 9000 },
    rim: { az: 155, el: 10, intensity: 0.9, kelvin: 2800, gel: '#ffb070' },
    ambient: 0.1, roughness: 0.55, spec: 0.15,
  },
  {
    name: '🕵️ Film noir',
    key: { az: -35, el: 55, intensity: 1.35, softness: 0.2, kelvin: 4800 },
    fill: { az: 60, el: 0, intensity: 0.02, softness: 1, kelvin: 5600 },
    rim: { az: 170, el: 35, intensity: 0.5, kelvin: 6000 },
    ambient: 0.02, roughness: 0.4, spec: 0.25,
  },
  {
    name: '👻 Horror',
    key: { az: 0, el: -50, intensity: 1.2, softness: 0.5, kelvin: 6500, gel: '#b0ffb8' },
    fill: { az: 90, el: 0, intensity: 0.02, softness: 1, kelvin: 6500 },
    rim: { az: 180, el: 0, intensity: 0.3, kelvin: 6500, gel: '#60ff80' },
    ambient: 0.02,
  },
  {
    name: '🏖 Beach midday',
    key: { az: -35, el: 60, intensity: 1.15, softness: 0.45, kelvin: 5800 },
    fill: { az: 40, el: -10, intensity: 0.45, softness: 1, kelvin: 7500, gel: '#d8f0ff' },
    rim: { az: 160, el: 40, intensity: 0.6, kelvin: 5600 },
    ambient: 0.2, roughness: 0.45, spec: 0.2,
  },
  {
    name: '🛋 Warm lamp',
    key: { az: -55, el: 15, intensity: 1.0, softness: 0.85, kelvin: 2900 },
    fill: { az: 60, el: 10, intensity: 0.18, softness: 1, kelvin: 4200 },
    rim: { az: 150, el: 25, intensity: 0.4, kelvin: 3000 },
    ambient: 0.08, roughness: 0.55, spec: 0.12,
  },
];

export function drawMatcapPreview(canvas: HTMLCanvasElement, data: Float32Array): void {
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(MATCAP_SIZE, MATCAP_SIZE);
  for (let i = 0; i < MATCAP_SIZE * MATCAP_SIZE; i++) {
    for (let c = 0; c < 3; c++) img.data[i * 4 + c] = 255 * linearToSrgb(Math.min(1, data[i * 4 + c]! * 0.5));
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

export const PANEL_CSS = `
  html, body { margin: 0; height: 100%; background: #111; color: #ddd; font: 12px/1.4 system-ui, sans-serif; }
  #app { display: grid; grid-template-columns: minmax(0, 1fr) 340px; height: 100vh; overflow: hidden; }
  #view { position: relative; display: flex; align-items: center; justify-content: center; overflow: hidden; min-width: 0; }
  canvas#gfx { max-width: 100%; max-height: 100%; background: #000; }
  #hud { position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,.65); padding: 8px 10px; border-radius: 6px; white-space: pre; font-family: ui-monospace, monospace; }
  #panel { width: 340px; min-width: 340px; max-width: 340px; box-sizing: border-box; overflow-y: auto; overflow-x: hidden; overflow-wrap: anywhere; padding: 10px 12px; background: #1a1a1a; border-left: 1px solid #333; }
  #panel * { max-width: 100%; box-sizing: border-box; }
  fieldset { border: 1px solid #333; border-radius: 6px; margin: 0 0 10px; padding: 6px 8px; }
  legend { color: #9cf; padding: 0 4px; }
  .row { display: grid; grid-template-columns: 92px 1fr 44px; align-items: center; gap: 6px; margin: 3px 0; }
  .row input[type=range] { width: 100%; }
  .row output { text-align: right; font-family: ui-monospace, monospace; color: #aaa; }
  select { width: 100%; background: #222; color: #ddd; border: 1px solid #444; }
  #matcap { width: 128px; height: 128px; image-rendering: pixelated; border: 1px solid #333; display: block; margin: 4px auto; }
  #err { position: absolute; bottom: 10px; left: 10px; color: #f77; white-space: pre-wrap; max-width: 90%; }
  video { display: none; }
  .ok { color: #7f7; } .bad { color: #f77; }
`;
