// Usage: node scripts/inspect-models.mjs models/a.tflite,models/b.tflite
import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new', args: ['--enable-unsafe-webgpu', '--use-angle=metal'],
});
const page = await browser.newPage();
const logs = [];
page.on('console', (m) => { const t = m.text(); if (/delegat|WebGPU-based|fallback|unsupported|not supported/i.test(t)) logs.push(t.slice(0, 200)); });
await page.goto(`http://localhost:5173/inspect.html?m=${process.argv[2]}`);
await page.waitForSelector('#out[data-done]', { timeout: 60000 });
console.log(await page.$eval('#out', (e) => e.textContent));
console.log(logs.join('\n'));
await browser.close();
