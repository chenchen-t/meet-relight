// Headless smoke test: opens a demo page in system Chrome with WebGPU + fake
// camera, waits, prints console errors + HUD text, saves a screenshot.
// Usage: node scripts/smoke.mjs [url] [screenshot.png] [waitMs]
import puppeteer from 'puppeteer-core';

const url = process.argv[2] ?? 'http://localhost:5173/';
const shot = process.argv[3] ?? 'smoke.png';
const waitMs = Number(process.argv[4] ?? 8000);

const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new',
  args: [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan,WebGPU',
    '--use-angle=metal',
    '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream',
    '--autoplay-policy=no-user-gesture-required',
    ...(process.env.FAKE_VIDEO ? [`--use-file-for-fake-video-capture=${process.env.FAKE_VIDEO}`] : []),
  ],
});
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 800 });
const logs = [];
page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
await page.goto(url, { waitUntil: 'load' });
await new Promise((r) => setTimeout(r, waitMs));
const hud = await page.$eval('#hud', (e) => e.textContent).catch(() => '(no #hud)');
const err = await page.$eval('#err', (e) => e.textContent).catch(() => '');
const gr = await page.$eval('#gr', (e) => e.textContent).catch(() => '');
if (gr) console.log('--- greenroom ---\n' + gr);
await page.screenshot({ path: shot });
console.log('--- HUD ---\n' + hud);
if (err) console.log('--- #err ---\n' + err);
console.log('--- console ---\n' + logs.filter((l) => !l.includes('[debug]')).slice(0, 40).join('\n'));
await browser.close();
