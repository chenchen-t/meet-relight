// Generic capture: node cap.mjs <y4m> <url> <out.png> [clicks comma-sep] [uncheck label prefix comma-sep] [clipX,clipY,w,h]
import puppeteer from 'puppeteer-core';
const [video, url, out, clicks = '', unchecks = '', clip = '0,130,960,540'] = process.argv.slice(2);
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new',
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=metal', '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream', '--autoplay-policy=no-user-gesture-required', `--use-file-for-fake-video-capture=${video}`],
});
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 800 });
page.on('pageerror', (e) => console.log('pageerror', e.message));
page.on('console', (m) => { const t = m.text(); if (m.type() === 'error' && !/INFO|WARNING/.test(t)) console.log('console', t); if (t.startsWith('[cap]')) console.log(t); });
await page.goto(url, { waitUntil: 'load' });
await new Promise((r) => setTimeout(r, 8000));
await page.evaluate(() => { const h = document.getElementById('hud'); if (h) h.style.display = 'none'; });
for (const c of clicks.split(',').filter(Boolean)) {
  await page.evaluate((c) => [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === c || b.textContent.includes(c))?.click(), c);
  await new Promise((r) => setTimeout(r, 800));
}
for (const u of unchecks.split(',').filter(Boolean)) {
  await page.evaluate((u) => {
    const l = [...document.querySelectorAll('label')].find((x) => x.textContent.trim().startsWith(u));
    const i = l?.querySelector('input'); if (i && i.checked) { i.checked = false; i.dispatchEvent(new Event('change')); }
  }, u);
}
await new Promise((r) => setTimeout(r, 1500));
const [x, y, width, height] = clip.split(',').map(Number);
await page.screenshot({ path: out, clip: { x, y, width, height } });
await browser.close();
