// E2E: load Gemma 4 in the live page (headless Chrome, WebGPU), send prompts, print the assistant log.
// node scripts/gemma_e2e.mjs <y4m> [prompts separated by |] [refImage]
import puppeteer from 'puppeteer-core';
import { fileURLToPath } from 'node:url';
const [video, prompts = 'film noir detective|warmer', refImage] = process.argv.slice(2);
// Persistent profile so the multi-GB model stays in the Cache API between runs.
const profile = process.env.CHROME_PROFILE ?? fileURLToPath(new URL('../.cache/chrome-gemma-profile', import.meta.url));
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new',
  userDataDir: profile,
  protocolTimeout: 1_800_000,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=metal', '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${video}`],
});
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 900 });
page.on('pageerror', (e) => console.log('pageerror', e.message));
page.on('console', (m) => { if (m.type() === 'error' && !/INFO|WARNING|gl_context/.test(m.text())) console.log('console', m.text().slice(0, 300)); });
await page.goto('http://localhost:5173/live.html?mode=0', { waitUntil: 'load' });
await new Promise((r) => setTimeout(r, 6000));
// Main-thread responsiveness monitor: longest rAF gap + frames rendered (camera → relight loop health).
await page.evaluate(() => {
  const w = window; w.__gap = 0; w.__frames = 0; let last = performance.now();
  const tick = (t) => { w.__gap = Math.max(w.__gap, t - last); last = t; w.__frames++; requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
});
const mon = async (label, t) => { const r = await page.evaluate(() => { const o = { gap: window.__gap, frames: window.__frames }; window.__gap = 0; window.__frames = 0; return o; }); console.log(`[main thread] ${label}: longest frame gap ${r.gap.toFixed(0)} ms, ${(r.frames / t).toFixed(0)} rAF/s`); };
await mon('baseline (2 s)', 1); await new Promise((r) => setTimeout(r, 2000)); await mon('baseline', 2);
const t0 = Date.now();
await page.evaluate(() => [...document.querySelectorAll('button')].find((b) => b.textContent === 'Load').click());
let last = '';
for (;;) {
  const st = await page.evaluate(() => [...document.querySelectorAll('fieldset')].find((f) => f.textContent.includes('Gemma 4'))?.querySelector('.gm-status')?.textContent ?? '');
  if (st !== last && (/ready|failed|HTTP/.test(st) || Math.random() < 0.05)) console.log(`[${((Date.now() - t0) / 1000).toFixed(0)}s] ${st}`);
  last = st;
  if (/ready/.test(st)) { await mon('during load', (Date.now() - t0) / 1000); break; }
  const err = await page.evaluate(() => document.querySelector('.gm-log .e')?.textContent);
  if (err) { console.log('ERROR', err); process.exit(1); }
  await new Promise((r) => setTimeout(r, 2000));
}
const runPrompt = async (label, fn) => {
  const before = await page.evaluate(() => document.querySelectorAll('.gm-log > div').length);
  await fn();
  const t1 = Date.now();
  for (;;) {
    await new Promise((r) => setTimeout(r, 1000));
    const lines = await page.evaluate(() => [...document.querySelectorAll('.gm-log > div')].map((d) => d.textContent));
    const fresh = lines.slice(before);
    if (fresh.some((l) => /^\d+\.\d s$|no change made/.test(l.trim())) || Date.now() - t1 > 240_000) {
      console.log(`--- ${label} (${((Date.now() - t1) / 1000).toFixed(1)} s)\n${fresh.join('\n')}`);
      await mon(`during "${label}"`, (Date.now() - t1) / 1000);
      break;
    }
  }
};
for (const q of prompts.split('|').filter(Boolean)) {
  await runPrompt(q, () => page.evaluate((q) => {
    const fs = [...document.querySelectorAll('fieldset')].find((f) => f.textContent.includes('Gemma 4'));
    const input = fs.querySelector('input.gm-in');
    input.value = q;
    [...fs.querySelectorAll('button')].find((b) => b.textContent === '➤').click();
  }, q));
  await page.screenshot({ path: `${profile}/../gemma_${q.replace(/\W+/g, '_').slice(0, 30)}.png` });
}
if (refImage) {
  await runPrompt(`reference ${refImage}`, async () => {
    const fs = await page.$$('fieldset');
    for (const f of fs) {
      if ((await f.evaluate((e) => e.textContent)).includes('Gemma 4')) {
        const input = await f.$('input[type=file]');
        await input.uploadFile(refImage);
      }
    }
  });
  await page.screenshot({ path: `${profile}/../gemma_reference.png` });
}
const rig = await page.evaluate(() => [...document.querySelectorAll('fieldset')].filter((f) => /Key light|Fill light|Rim/.test(f.querySelector('legend')?.textContent ?? ''))
  .map((f) => f.querySelector('legend').textContent + ': ' + [...f.querySelectorAll('.row')].map((r) => `${r.querySelector('span').textContent}=${r.querySelector('output')?.textContent}`).join(' ')).join('\n'));
console.log('--- rig\n' + rig);
await browser.close();
