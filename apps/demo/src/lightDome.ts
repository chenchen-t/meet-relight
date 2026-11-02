import { lightDirection, type TargetProfile, type Light } from '@relight/shared-schema';
import { targetShading, prepareLights, linearToSrgb, type Vec3 } from '@relight/relight-shader';

/**
 * Interactive hemisphere "light dome": shows where each virtual light sits around
 * the subject (camera-space az/el), with a head preview shaded by the target
 * lighting. Drag a light handle to move it; `onMove(role, az, el)` fires.
 *
 * View: orthographic, from slightly above the camera (tilt), so lights in front
 * of the subject appear lower and lights behind appear higher on the dome.
 */
export interface LightDomeOptions {
  size?: number;
  tiltDeg?: number;
  onMove: (role: Light['role'], azDeg: number, elDeg: number) => void;
}

const ROLE_STYLE: Record<Light['role'], { label: string; ring: string }> = {
  key: { label: 'KEY', ring: '#ffd27a' },
  fill: { label: 'FILL', ring: '#9cc9ff' },
  rim: { label: 'RIM', ring: '#d6a4ff' },
};

const HEAD_R = 0.26; // head radius relative to dome radius
const SKIN: Vec3 = [0.62, 0.42, 0.34];

export class LightDome {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private size: number;
  private tilt: number;
  private profile: TargetProfile | null = null;
  private mirror = true;
  private drag: Light['role'] | null = null;
  private hover: Light['role'] | null = null;
  private head: ImageData | null = null;
  private headDirty = true;
  private dpr = Math.min(2, window.devicePixelRatio || 1);

  constructor(container: HTMLElement, private opts: LightDomeOptions) {
    this.size = opts.size ?? 260;
    this.tilt = ((opts.tiltDeg ?? 24) * Math.PI) / 180;
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'light-dome';
    this.canvas.width = this.size * this.dpr;
    this.canvas.height = this.size * this.dpr;
    this.canvas.style.width = `${this.size}px`;
    this.canvas.style.height = `${this.size}px`;
    container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d')!;
    this.canvas.addEventListener('pointerdown', this.onDown);
    this.canvas.addEventListener('pointermove', this.onMoveEvt);
    this.canvas.addEventListener('pointerup', this.onUp);
    this.canvas.addEventListener('pointercancel', this.onUp);
    this.canvas.addEventListener('pointerleave', () => { if (!this.drag) { this.hover = null; this.draw(); } });
  }

  set visible(v: boolean) {
    this.canvas.style.display = v ? 'block' : 'none';
  }

  update(profile: TargetProfile, mirror: boolean): void {
    this.profile = profile;
    if (mirror !== this.mirror) this.headDirty = true;
    this.mirror = mirror;
    this.headDirty = true;
    this.draw();
  }

  // ------------------------------------------------------------ projection
  private get R(): number {
    return this.size * 0.4;
  }
  private get cx(): number {
    return this.size / 2;
  }
  private get cy(): number {
    return this.size * 0.56;
  }

  /** Camera-space direction → dome screen point (+ depth toward viewer). */
  private project(d: Vec3): { x: number; y: number; depth: number } {
    const x = this.mirror ? -d[0] : d[0]; // selfie mirror: light appears where it lights the mirrored image
    const sy = d[1] * Math.cos(this.tilt) - d[2] * Math.sin(this.tilt);
    const depth = d[1] * Math.sin(this.tilt) + d[2] * Math.cos(this.tilt);
    return { x: this.cx + x * this.R, y: this.cy - sy * this.R, depth };
  }

  private projAzEl(az: number, el: number) {
    return this.project(lightDirection(az, el));
  }

  // ------------------------------------------------------------ drawing
  private polyline(points: { x: number; y: number; depth: number }[], front: boolean): void {
    const ctx = this.ctx;
    ctx.beginPath();
    let pen = false;
    for (const p of points) {
      const vis = front ? p.depth >= -0.02 : p.depth < 0.02;
      if (vis) {
        if (!pen) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
        pen = true;
      } else pen = false;
    }
    ctx.stroke();
  }

  private drawGrid(front: boolean): void {
    const ctx = this.ctx;
    ctx.lineWidth = 1;
    ctx.setLineDash(front ? [] : [3, 4]);
    // Latitude rings (elevation).
    for (const el of [0, 30, 60]) {
      ctx.strokeStyle = el === 0 ? (front ? 'rgba(160,200,255,.55)' : 'rgba(160,200,255,.22)') : front ? 'rgba(255,255,255,.28)' : 'rgba(255,255,255,.1)';
      const pts = [];
      for (let az = -180; az <= 180; az += 5) pts.push(this.projAzEl(az, el));
      this.polyline(pts, front);
    }
    // Meridians (azimuth) every 30°.
    ctx.strokeStyle = front ? 'rgba(255,255,255,.22)' : 'rgba(255,255,255,.08)';
    for (let az = -180; az < 180; az += 30) {
      const pts = [];
      for (let el = 0; el <= 90; el += 5) pts.push(this.projAzEl(az, el));
      this.polyline(pts, front);
    }
    ctx.setLineDash([]);
  }

  private renderHead(): ImageData {
    const px = Math.round(HEAD_R * this.R * 2 * this.dpr);
    const img = this.ctx.createImageData(px, px);
    if (!this.profile) return img;
    const rig = this.profile.lighting_rig;
    const prepared = prepareLights(rig, { sss: 1, gamma: 1 });
    const t: Vec3 = [0, 0, 0];
    for (let j = 0; j < px; j++) {
      for (let i = 0; i < px; i++) {
        const sx = ((i + 0.5) / px) * 2 - 1;
        const sy = 1 - ((j + 0.5) / px) * 2;
        const r2 = sx * sx + sy * sy;
        const k = (j * px + i) * 4;
        if (r2 > 1) continue;
        // View-space normal on the head sphere → camera space (undo tilt, undo mirror).
        const vz = Math.sqrt(1 - r2);
        const ny = sy * Math.cos(this.tilt) + vz * Math.sin(this.tilt);
        const nz = -sy * Math.sin(this.tilt) + vz * Math.cos(this.tilt);
        const nx = this.mirror ? -sx : sx;
        targetShading(rig, [nx, ny, nz], { sss: 1, gamma: 1 }, prepared, t);
        const a = Math.min(1, (1 - Math.sqrt(r2)) * 12); // AA edge
        for (let c = 0; c < 3; c++) img.data[k + c] = 255 * linearToSrgb(Math.min(1, SKIN[c]! * t[c]! * 1.1));
        img.data[k + 3] = 255 * a;
      }
    }
    return img;
  }

  private lightHandles(): { role: Light['role']; light: Light; p: { x: number; y: number; depth: number } }[] {
    if (!this.profile) return [];
    return this.profile.lighting_rig.lights.map((l) => ({ role: l.role, light: l, p: this.projAzEl(l.azimuth_deg, l.elevation_deg) }));
  }

  private drawLight(h: { role: Light['role']; light: Light; p: { x: number; y: number; depth: number } }, behindHead: boolean): void {
    const ctx = this.ctx;
    const { light: l, p } = h;
    const style = ROLE_STYLE[l.role];
    const rgb = l.color_rgb.map((v) => Math.round(255 * Math.min(1, v))).join(',');
    const back = p.depth < 0;
    const alpha = back ? 0.55 : 1;
    // Beam toward the head.
    ctx.strokeStyle = `rgba(${rgb},${0.35 * alpha})`;
    ctx.setLineDash(behindHead ? [2, 4] : []);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(this.cx, this.cy);
    ctx.stroke();
    ctx.setLineDash([]);
    // Softness halo + intensity-sized core.
    const core = 4 + 5 * Math.min(2, l.intensity);
    const halo = core + 4 + 16 * l.softness;
    const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, halo);
    g.addColorStop(0, `rgba(${rgb},${0.9 * alpha})`);
    g.addColorStop(core / halo, `rgba(${rgb},${0.55 * alpha})`);
    g.addColorStop(1, `rgba(${rgb},0)`);
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(p.x, p.y, halo, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = this.drag === h.role || this.hover === h.role ? 2.5 : 1.5;
    ctx.strokeStyle = style.ring;
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.arc(p.x, p.y, core, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = style.ring;
    ctx.font = '600 9px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(style.label, p.x, p.y - core - 4);
    if (this.drag === h.role) {
      ctx.fillStyle = '#fff';
      ctx.fillText(`az ${Math.round(l.azimuth_deg)}° el ${Math.round(l.elevation_deg)}°`, p.x, p.y + core + 12);
    }
    ctx.globalAlpha = 1;
  }

  draw(): void {
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.size, this.size);
    // Panel background.
    ctx.fillStyle = 'rgba(12,14,20,.72)';
    ctx.beginPath();
    ctx.roundRect(0, 0, this.size, this.size, 10);
    ctx.fill();
    ctx.fillStyle = '#9cf';
    ctx.font = '600 11px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Light dome — drag lights', 10, 16);

    const handles = this.lightHandles();
    // Back half: grid + lights behind the dome's visible front.
    this.drawGrid(false);
    for (const h of handles) if (h.p.depth < 0) this.drawLight(h, true);

    // Floor shadow + head.
    ctx.fillStyle = 'rgba(0,0,0,.35)';
    ctx.beginPath();
    ctx.ellipse(this.cx, this.cy + HEAD_R * this.R * 0.95, HEAD_R * this.R * 0.9, HEAD_R * this.R * 0.25, 0, 0, Math.PI * 2);
    ctx.fill();
    if (this.headDirty) {
      this.head = this.renderHead();
      this.headDirty = false;
    }
    if (this.head) {
      const off = new OffscreenCanvas(this.head.width, this.head.height);
      off.getContext('2d')!.putImageData(this.head, 0, 0);
      const s = HEAD_R * this.R;
      ctx.drawImage(off, this.cx - s, this.cy - s, 2 * s, 2 * s);
    }

    // Camera marker (front).
    const cam = this.project([0, 0, 1.18]);
    ctx.fillStyle = 'rgba(255,255,255,.75)';
    ctx.fillRect(cam.x - 9, cam.y - 5, 18, 10);
    ctx.beginPath();
    ctx.arc(cam.x, cam.y, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#222';
    ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,.6)';
    ctx.font = '9px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('camera', cam.x, cam.y + 16);

    this.drawGrid(true);
    for (const h of handles) if (h.p.depth >= 0) this.drawLight(h, false);
  }

  // ------------------------------------------------------------ interaction
  private local(e: PointerEvent): [number, number] {
    const r = this.canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  private pick(x: number, y: number): Light['role'] | null {
    let best: Light['role'] | null = null;
    let bestD = 16;
    // Prefer front-most handle on ties.
    for (const h of this.lightHandles().sort((a, b) => b.p.depth - a.p.depth)) {
      const d = Math.hypot(h.p.x - x, h.p.y - y);
      if (d < bestD) {
        bestD = d;
        best = h.role;
      }
    }
    return best;
  }

  /** Find (az, el) whose projection is closest to (x, y), preferring continuity with the current position. */
  private solve(x: number, y: number, role: Light['role']): [number, number] {
    const cur = this.profile!.lighting_rig.lights.find((l) => l.role === role)!;
    const curDir = lightDirection(cur.azimuth_deg, cur.elevation_deg);
    let best: [number, number] = [cur.azimuth_deg, cur.elevation_deg];
    let bestCost = Infinity;
    for (let el = -30; el <= 90; el += 2) {
      for (let az = -180; az < 180; az += 2) {
        const d = lightDirection(az, el);
        const p = this.project(d);
        const screen = Math.hypot(p.x - x, p.y - y);
        const cont = Math.acos(Math.max(-1, Math.min(1, d[0] * curDir[0] + d[1] * curDir[1] + d[2] * curDir[2])));
        const cost = screen + cont * 12;
        if (cost < bestCost) {
          bestCost = cost;
          best = [az, el];
        }
      }
    }
    return best;
  }

  private onDown = (e: PointerEvent) => {
    const [x, y] = this.local(e);
    this.drag = this.pick(x, y);
    if (this.drag) {
      this.canvas.setPointerCapture(e.pointerId);
      e.preventDefault();
    }
  };

  private onMoveEvt = (e: PointerEvent) => {
    const [x, y] = this.local(e);
    if (!this.drag) {
      const h = this.pick(x, y);
      if (h !== this.hover) {
        this.hover = h;
        this.canvas.style.cursor = h ? 'grab' : 'default';
        this.draw();
      }
      return;
    }
    const [az, el] = this.solve(x, y, this.drag);
    this.opts.onMove(this.drag, az, el);
  };

  private onUp = (e: PointerEvent) => {
    if (this.drag) this.canvas.releasePointerCapture(e.pointerId);
    this.drag = null;
    this.draw();
  };
}
