// Panel layout: a compact always-visible quick bar (view mode, strength) + tabs that group the many
// control groups by intent, with collapsible sections. Works on the existing fieldsets (no control code
// changes): sections are matched by their legend text.

const CSS = `
#panel { padding: 0 !important; display: flex; flex-direction: column; }
.ly-quick { position: sticky; top: 0; z-index: 2; background: #1d1d1d; border-bottom: 1px solid #333; padding: 10px 12px 8px; }
.ly-title { font-weight: 700; font-size: 14px; margin-bottom: 8px; display:flex; justify-content:space-between; align-items:center; }
.ly-title small { font-weight: 400; color: #8a8a8a; font-size: 11px; }
.ly-sub { font-size: 11px; color: #9cf; margin: -6px 0 8px; }
.ly-seg { display: flex; background: #111; border-radius: 8px; padding: 3px; gap: 3px; margin-bottom: 8px; }
.ly-seg button { flex: 1; margin: 0 !important; padding: 6px 4px !important; background: transparent !important; color: #bbb !important; border-radius: 6px !important; font-size: 12px; }
.ly-seg button.on { background: #2f6fdd !important; color: #fff !important; }
.ly-quick .row { margin: 2px 0; }
.ly-quick label { font-size: 12px; }
.ly-tabs { display: flex; gap: 2px; padding: 8px 12px 0; position: sticky; top: var(--ly-quick-h, 120px); z-index: 1; background: #1a1a1a; border-bottom: 1px solid #333; }
.ly-tabs button { flex: 1; white-space: nowrap; margin: 0 !important; padding: 7px 2px !important; background: transparent !important; color: #999 !important; border-radius: 6px 6px 0 0 !important; font-size: 12px; border-bottom: 2px solid transparent !important; }
.ly-tabs button.on { color: #fff !important; border-bottom-color: #4c8dff !important; background: #222 !important; }
.ly-page { display: none; padding: 8px 12px 24px; }
.ly-page.on { display: block; }
.ly-page fieldset { border: 1px solid #2c2c2c; border-radius: 8px; margin: 0 0 10px; padding: 6px 10px 8px; background: #1e1e1e; }
.ly-page legend { cursor: pointer; user-select: none; padding: 0 6px; font-weight: 600; color: #ddd; }
.ly-page legend::before { content: '▾ '; color: #777; }
.ly-page fieldset.ly-closed > :not(legend) { display: none !important; }
.ly-page fieldset.ly-closed legend::before { content: '▸ '; }
.ly-hint { color: #888; font-size: 11px; margin: 2px 0 10px; }
#hud { cursor: pointer; }
`;

export interface TabSpec {
  id: string;
  label: string;
  hint?: string;
  /** Legend regexes of the sections on this tab, in display order. */
  sections: RegExp[];
  /** Sections that start collapsed. */
  collapsed?: RegExp[];
}

export interface LayoutOptions {
  panel: HTMLElement;
  title: string;
  subtitle?: string;
  /** View-mode buttons: label → value of the select labelled `modeLabel`. */
  modes: Array<[string, string]>;
  modeLabel: string;
  /** Row labels (span text / checkbox label prefix) moved from their section into the quick bar. */
  quickRows: string[];
  tabs: TabSpec[];
}

const legendOf = (fs: HTMLFieldSetElement) => fs.querySelector('legend')?.textContent?.trim() ?? '';

export function applyLayout(o: LayoutOptions): { selectTab: (id: string) => void } {
  document.head.insertAdjacentHTML('beforeend', `<style>${CSS}</style>`);
  const { panel } = o;
  const fieldsets = [...panel.querySelectorAll(':scope > fieldset')] as HTMLFieldSetElement[];

  // ---- quick bar
  const quick = document.createElement('div');
  quick.className = 'ly-quick';
  quick.innerHTML = `<div class="ly-title">${o.title}<small>click the HUD to expand stats</small></div>` +
    (o.subtitle ? `<div class="ly-sub">${o.subtitle}</div>` : '');
  const modeRow = [...panel.querySelectorAll('.row')].find((r) => r.querySelector('span')?.textContent === o.modeLabel);
  const modeSel = modeRow?.querySelector('select') as HTMLSelectElement | undefined;
  if (modeSel) {
    const seg = document.createElement('div');
    seg.className = 'ly-seg';
    const btns: HTMLButtonElement[] = [];
    const sync = () => btns.forEach((b) => b.classList.toggle('on', b.dataset.v === modeSel.value));
    for (const [label, v] of o.modes) {
      const b = document.createElement('button');
      b.textContent = label;
      b.dataset.v = v;
      b.onclick = () => {
        modeSel.value = v;
        modeSel.dispatchEvent(new Event('change'));
        sync();
      };
      btns.push(b);
      seg.appendChild(b);
    }
    modeSel.addEventListener('change', sync);
    sync();
    quick.appendChild(seg);
  }
  for (const label of o.quickRows) {
    const row = [...panel.querySelectorAll('.row, label')].find((r) =>
      (r.querySelector(':scope > span')?.textContent ?? r.textContent ?? '').trim().startsWith(label));
    if (row) quick.appendChild(row);
  }

  // ---- tabs
  const tabBar = document.createElement('div');
  tabBar.className = 'ly-tabs';
  const pages = new Map<string, HTMLDivElement>();
  const tabBtns = new Map<string, HTMLButtonElement>();
  const used = new Set<HTMLFieldSetElement>();
  for (const t of o.tabs) {
    const page = document.createElement('div');
    page.className = 'ly-page';
    if (t.hint) page.insertAdjacentHTML('beforeend', `<div class="ly-hint">${t.hint}</div>`);
    for (const re of t.sections) {
      for (const fs of fieldsets) {
        if (used.has(fs) || !re.test(legendOf(fs))) continue;
        used.add(fs);
        page.appendChild(fs);
        if (t.collapsed?.some((c) => c.test(legendOf(fs)))) fs.classList.add('ly-closed');
      }
    }
    pages.set(t.id, page);
    const b = document.createElement('button');
    b.textContent = t.label;
    b.onclick = () => selectTab(t.id);
    tabBtns.set(t.id, b);
    tabBar.appendChild(b);
  }
  // Anything unmatched goes to the last tab (nothing gets lost).
  const last = pages.get(o.tabs[o.tabs.length - 1]!.id)!;
  for (const fs of fieldsets) if (!used.has(fs)) { last.appendChild(fs); fs.classList.add('ly-closed'); }
  for (const fs of fieldsets) {
    fs.querySelector('legend')?.addEventListener('click', () => fs.classList.toggle('ly-closed'));
  }

  panel.prepend(quick, tabBar, ...pages.values());
  const selectTab = (id: string) => {
    for (const [k, p] of pages) p.classList.toggle('on', k === id);
    for (const [k, b] of tabBtns) b.classList.toggle('on', k === id);
    try { localStorage.setItem('meet-relight-tab', id); } catch { /* private mode */ }
  };
  const saved = (() => { try { return localStorage.getItem('meet-relight-tab'); } catch { return null; } })();
  selectTab(saved && pages.has(saved) ? saved : o.tabs[0]!.id);
  const setH = () => document.documentElement.style.setProperty('--ly-quick-h', `${quick.offsetHeight}px`);
  setH();
  new ResizeObserver(setH).observe(quick);
  return { selectTab };
}
