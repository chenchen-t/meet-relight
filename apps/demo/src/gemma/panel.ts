// "Gemma 4 lighting assistant" panel. Gemma runs in a worker (engine.ts), so loading / generating never
// blocks the camera. Features:
//  (1) describe a look in words → set_lighting,
//  (2) small spoken / typed adjustments → adjust_light,
//  (3) a reference photo → reproduce its lighting (vision + measured hints).
import { GemmaClient, gemmaCached, gemmaSupported, GEMMA_MODELS, toPromptImage } from './engine.ts';
import { referencePrompt, runTurn, toolPrompt, type LightingTarget } from './agent.ts';
import { listen, listening, speechSupported, stopListening } from './speech.ts';
import { estimateLightingFromBackground, measureLitSide } from '../bgLight.ts';

const CSS = `
.gm-head { display:flex; align-items:center; gap:6px; margin:2px 0 6px; }
.gm-pill { font-size:11px; padding:2px 8px; border-radius:10px; background:#333; color:#bbb; white-space:nowrap; }
.gm-pill.ready { background:#1e4d2b; color:#9fe0ae; } .gm-pill.busy { background:#2a3d66; color:#bcd2ff; } .gm-pill.err { background:#5a2323; color:#ffb4b4; }
.gm-head select { flex:1; min-width:0; }
.gm-btn { width:auto !important; margin:0 !important; padding:5px 9px !important; white-space:nowrap; border-radius:6px !important; }
.gm-btn.ghost { background:#2c2c2c !important; }
.gm-bar { height:3px; background:#2a2a2a; border-radius:2px; overflow:hidden; margin:0 0 6px; }
.gm-bar > div { height:100%; width:0; background:#4c8dff; transition:width .2s; }
.gm-status { font-size:11px; color:#8fa3b8; margin-bottom:6px; min-height:14px; }
.gm-box { display:flex; gap:4px; background:#141414; border:1px solid #3a3a3a; border-radius:8px; padding:4px; }
.gm-box:focus-within { border-color:#4c8dff; }
.gm-in { flex:1; min-width:0; background:transparent; color:#eee; border:0; outline:0; padding:5px; font:13px system-ui; }
.gm-chips { display:flex; flex-wrap:wrap; gap:4px; margin:6px 0; }
.gm-chip { width:auto !important; margin:0 !important; padding:3px 9px !important; font-size:11px; background:#2b2b2b !important; color:#ccc !important; border-radius:12px !important; }
.gm-chip:hover { background:#36507a !important; color:#fff !important; }
.gm-log { max-height:200px; overflow-y:auto; display:flex; flex-direction:column; gap:4px; margin-top:4px; }
.gm-msg { font:12px/1.4 system-ui; padding:5px 8px; border-radius:8px; max-width:92%; white-space:pre-wrap; }
.gm-msg.u { align-self:flex-end; background:#2f6fdd; color:#fff; }
.gm-msg.t { align-self:flex-start; background:#2a2a2a; color:#e6e6e6; }
.gm-msg.c { align-self:flex-start; background:transparent; color:#8fd19e; font:11px ui-monospace,monospace; padding:0 8px; }
.gm-msg.e { align-self:flex-start; background:#3d1f1f; color:#ffb4b4; }
.gm-msg.s { align-self:center; color:#888; font-size:11px; padding:0; }
button:disabled { opacity:.45; cursor:default; }
`;

export function createGemmaPanel(parent: HTMLElement, target: LightingTarget, assetBase: string): void {
  document.head.insertAdjacentHTML('beforeend', `<style>${CSS}</style>`);
  const el = <K extends keyof HTMLElementTagNameMap>(tag: K, cls = '', text = '') => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text) e.textContent = text;
    return e;
  };
  const client = new GemmaClient(assetBase);

  // ---- header: state pill · model · load
  const head = el('div', 'gm-head');
  const pill = el('span', 'gm-pill', 'not loaded');
  const modelSel = el('select');
  for (const m of GEMMA_MODELS) modelSel.add(new Option(`${m.name} · ${(m.bytes / 1e9).toFixed(1)} GB`, m.id));
  const loadBtn = el('button', 'gm-btn', 'Load');
  head.append(pill, modelSel, loadBtn);
  const bar = el('div', 'gm-bar');
  const barFill = el('div');
  bar.appendChild(barFill);
  const status = el('div', 'gm-status');

  // ---- chat box
  const box = el('div', 'gm-box');
  const input = el('input', 'gm-in');
  input.placeholder = 'Describe a look… e.g. "Wong Kar-wai neon"';
  const micBtn = el('button', 'gm-btn ghost', '🎤');
  micBtn.title = speechSupported() ? 'Speak a request' : 'Speech recognition not supported in this browser';
  micBtn.disabled = !speechSupported();
  const refBtn = el('button', 'gm-btn ghost', '🖼');
  refBtn.title = 'Match the lighting of a reference photo';
  const sendBtn = el('button', 'gm-btn', '➤');
  sendBtn.title = 'Send (Enter)';
  const file = el('input');
  file.type = 'file';
  file.accept = 'image/*';
  file.style.display = 'none';
  box.append(input, micBtn, refBtn, sendBtn, file);

  const chips = el('div', 'gm-chips');
  for (const c of ['Wong Kar-wai neon', 'soft window light from the left', 'golden hour', 'film noir', 'warmer', 'less shadow', 'brighter']) {
    const b = el('button', 'gm-chip', c);
    b.onclick = () => { input.value = c; void ask(c); };
    chips.appendChild(b);
  }
  const logEl = el('div', 'gm-log');
  parent.append(head, bar, status, box, chips, logEl);

  const log = (cls: string, text: string) => {
    const line = el('div', `gm-msg ${cls}`, text);
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
    return line;
  };
  const setPill = (text: string, cls = '') => { pill.textContent = text; pill.className = `gm-pill ${cls}`; };

  // ---- state
  let busy = false;
  let generating = false;
  const refresh = () => {
    const ready = client.ready;
    loadBtn.disabled = busy || ready;
    loadBtn.textContent = ready ? '✓' : busy ? '…' : 'Load';
    sendBtn.textContent = generating ? '■' : '➤';
    sendBtn.title = generating ? 'Stop' : 'Send (Enter)';
    refBtn.disabled = generating;
  };
  const showCacheState = async () => {
    if (!gemmaSupported()) { setPill('no WebGPU', 'err'); status.textContent = 'WebGPU / Cache API not available in this browser.'; return; }
    const cached = await gemmaCached(modelSel.value);
    if (!client.ready) {
      setPill(cached ? 'cached' : 'not loaded');
      status.textContent = cached
        ? 'Downloaded before — loads in a few seconds (in a background worker; video keeps running).'
        : `First load downloads ${(GEMMA_MODELS.find((m) => m.id === modelSel.value)!.bytes / 1e9).toFixed(1)} GB once (background worker; video keeps running).`;
    }
  };

  const ensure = async (): Promise<boolean> => {
    if (client.ready) return true;
    busy = true;
    refresh();
    setPill('loading', 'busy');
    try {
      await client.load(modelSel.value, (text, f) => {
        status.textContent = text;
        if (f !== undefined) {
          barFill.style.width = `${(f * 100).toFixed(1)}%`;
          setPill(`${Math.round(f * 100)}%`, 'busy');
        }
      });
      setPill('ready', 'ready');
      bar.style.visibility = 'hidden';
      return true;
    } catch (e) {
      setPill('error', 'err');
      log('e', `Load failed: ${e instanceof Error ? e.message : e}`);
      return false;
    } finally {
      busy = false;
      refresh();
    }
  };
  loadBtn.onclick = () => void ensure();
  modelSel.onchange = () => { bar.style.visibility = 'visible'; barFill.style.width = '0'; void showCacheState(); refresh(); };

  const turn = async (label: string, prompt: string, image: ImageBitmap | null) => {
    if (generating) return;
    if (!(await ensure())) return;
    generating = true;
    refresh();
    log('u', label);
    const thinking = log('s', 'thinking…');
    const t0 = performance.now();
    try {
      const r = await runTurn(
        (onPartial) => client.turn(prompt, image, onPartial),
        target,
        {
          onCall: (c, res) => {
            const args = c.name === 'set_lighting' ? '' : ` ${JSON.stringify(c.arguments)}`;
            log(res instanceof Error ? 'e' : 'c', `⚙ ${c.name}${args} → ${res instanceof Error ? res.message : res}`);
          },
          onText: (t) => log('t', t),
        },
      );
      thinking.textContent = `${((performance.now() - t0) / 1000).toFixed(1)} s`;
      if (!r.calls) log('s', 'no change made — try rephrasing');
    } catch (e) {
      thinking.remove();
      log('e', e instanceof Error ? e.message : String(e));
    } finally {
      generating = false;
      refresh();
    }
  };

  // (1) + (2): words → lighting
  const ask = (text: string) => {
    const q = text.trim();
    if (!q) return Promise.resolve();
    input.value = '';
    return turn(q, `${toolPrompt(target)}\nUser: ${q}`, null);
  };
  sendBtn.onclick = () => (generating ? client.cancel() : void ask(input.value));
  input.onkeydown = (e) => { if (e.key === 'Enter') void ask(input.value); };

  // (2) voice
  micBtn.onclick = async () => {
    if (listening()) { stopListening(); return; }
    micBtn.textContent = '⏺';
    try {
      const mode = await listen(navigator.language || 'en-US', (text, final) => {
        input.value = text;
        if (final) void ask(text);
      }, (err) => {
        micBtn.textContent = '🎤';
        if (err && err !== 'no-speech' && err !== 'aborted') log('e', `speech: ${err}`);
      });
      status.textContent = `Listening (${mode} speech recognition)…`;
    } catch (e) {
      micBtn.textContent = '🎤';
      log('e', `speech: ${e}`);
    }
  };

  // (3) reference image → lighting
  refBtn.onclick = () => file.click();
  file.onchange = async () => {
    const f = file.files?.[0];
    file.value = '';
    if (!f) return;
    const bmp = await createImageBitmap(f);
    // Numeric hints (colours, contrast, measured lit side). Small vision models often confuse
    // left/right, so the measured side also has the last word.
    const est = estimateLightingFromBackground(bmp, false);
    const lit = measureLitSide(bmp);
    const hints = `${lit.side ? `the subject is lit from the ${lit.side.toUpperCase()} of the picture (${lit.ratio.toFixed(1)}× brighter half); ` : ''}` +
      `light colour ${est.preset.key.gel}, opposite side colour ${est.preset.fill.gel}; contrast ${est.contrast.toFixed(1)} (1.5 flat … 6+ very contrasty)`;
    const img = await toPromptImage(bmp, 768);
    await turn(`🖼 ${f.name}${lit.side ? ` (lit from the ${lit.side})` : ''}`, referencePrompt(target, hints), img);
    if (lit.side) {
      const p = target.getPreset();
      const keyPicAz = target.mirrored() ? -p.key.az : p.key.az; // + = from the picture's right
      if ((lit.side === 'right') !== (keyPicAz > 0) && Math.abs(keyPicAz) > 5) {
        target.applyPreset({ ...p, name: 'Gemma', key: { ...p.key, az: -p.key.az }, fill: { ...p.fill, az: -p.fill.az }, rim: { ...p.rim, az: -p.rim.az } });
        log('c', `⚙ mirrored the rig so the key comes from the ${lit.side} (measured)`);
      }
    }
  };

  void showCacheState();
  refresh();
}
