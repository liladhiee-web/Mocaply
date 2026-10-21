// Frame-exact recordings of Prompt2Rig takes, rendered in an off-screen (headless) Chrome.
// Doesn't touch your screen: each frame is seeked, posed with the GVHMR motion and screenshotted.
//
//   node tools/capture.mjs --clip <video url>[,<more urls|url|prompt>] [--character michelle] [--out captures]
//
// --clip entries are URLs, or `url|label` to name each clip separately. --passes N plays each
// clip N times; --zoom a-b zooms onto --zoom-bone (default RightHand) between a and b seconds of
// the output, with --ramp seconds of ease in/out.
//
// Needs the web server (node server.mjs) and the HQ worker running. Writes one 2560×1440 30fps
// MP4 per clip: <out>/<nn>_<label>.mp4, plus <out>/takes.json.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import puppeteer from 'puppeteer-core';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (a.startsWith('--')) acc.push([a.slice(2), all[i + 1] && !all[i + 1].startsWith('--') ? all[i + 1] : true]);
  return acc;
}, []));
const W = Number(args.width || 2560), H = Number(args.height || 1440), FPS = 30;
const OUT = path.resolve(args.out || 'captures');
const URL = args.url || 'http://localhost:5177';
const CHROME = args.chrome || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const clips = String(args.clip || '').split(',').map((s) => s.trim()).filter(Boolean);
if (!clips.length) {
  console.error('usage: node tools/capture.mjs --clip <video url>[,<url|label>] [--prompt "..."] [--passes N] [--out captures]');
  process.exit(1);
}
const prompts = clips.map((c, i) => {
  const at = c.indexOf('|');
  const url = (at >= 0 ? c.slice(0, at) : c).trim();
  return { prompt: (at >= 0 ? c.slice(at + 1).trim() : args.prompt) || (clips.length > 1 ? `clip ${i + 1}` : 'existing clip'), url };
});

// App settings for the recording: GVHMR only, static camera.
const settings = { character: args.character || 'michelle', swap: false, cinematic: false, hq: true, showSkel: true };

const log = (...m) => console.log(new Date().toLocaleTimeString(), ...m);
const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '').slice(0, 48);

fs.mkdirSync(OUT, { recursive: true });
const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  userDataDir: fs.mkdtempSync(path.join(os.tmpdir(), 'p2r-capture-')),
  args: [`--window-size=${W},${H}`, '--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist',
    '--autoplay-policy=no-user-gesture-required', '--hide-scrollbars', '--mute-audio'],
  defaultViewport: { width: W, height: H, deviceScaleFactor: 1 },
  protocolTimeout: 900000,
});
const results = [];
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => log('page error:', e.message));
  await page.evaluateOnNewDocument((s) => { try { localStorage.setItem('p2r', JSON.stringify(s)); } catch {} }, settings); // sandboxed iframes can't
  await page.goto(URL, { waitUntil: 'networkidle2' });
  await page.waitForFunction(() => window.p2r?.rig && document.querySelector('#pill-hq').classList.contains('ok')
    && document.querySelector('#pill-pose').classList.contains('ok'), { timeout: 120000 }).catch(async (e) => {
    log('app not ready:', await page.evaluate(() => JSON.stringify({ p2r: !!window.p2r, rig: !!window.p2r?.rig,
      pills: [...document.querySelectorAll('.pill')].slice(0, 3).map((x) => x.className + ':' + x.textContent) })).catch(() => '?'));
    throw e;
  });
  log(`app ready (${settings.character}); ${prompts.length} clips`);

  const passes = Number(args.passes || 1);
  const [z0, z1] = args.zoom ? String(args.zoom).split('-').map(Number) : [Infinity, -Infinity];
  const ramp = Number(args.ramp || 1.1);
  const zoomAt = (t) => (t < z0 || t > z1 ? 0 : Math.min(1, (t - z0) / ramp, (z1 - t) / ramp));
  for (const [i, { prompt, url }] of prompts.entries()) {
    const name = `${String(i + 1).padStart(2, '0')}_${slug(prompt)}`;
    log(`#${i + 1} loading clip ${url}`);
    const id = await page.evaluate((u, p) => window.p2r.playUrl(u, true, undefined, p).id, url, prompt);
    const take = await page.waitForFunction((tid) => {
      const t = window.p2r.take(tid);
      return t && (t.status === 'error' || t.hq === 'error' || t.hq === 'done') && t;
    }, { timeout: 240000, polling: 500 }, id).then((h) => h.jsonValue());
    if (take.hq !== 'done') { log(`#${i + 1} failed (${take.status}/${take.hq}) — skipped`); continue; }

    const { duration, frames } = await page.evaluate((tid) => window.p2r.beginCapture(tid), id);
    const file = path.join(OUT, `${name}.mp4`);
    const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-c:v', 'mjpeg', '-r', String(FPS), '-i', '-',
      '-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', file], { stdio: ['pipe', 'inherit', 'inherit'] });
    const t0 = Date.now();
    for (let f = 0; f < frames * passes; f++) {
      const k = zoomAt(f / FPS);
      await page.evaluate((t, zk, bone, tt) => window.p2r.renderAt(t, { k: zk, bone, t: tt }), (f % frames) / FPS, k, args['zoom-bone'] || 'RightHand', f / FPS);
      const jpg = await page.screenshot({ type: 'jpeg', quality: 95, optimizeForSpeed: true });
      if (!ff.stdin.write(jpg)) await new Promise((r) => ff.stdin.once('drain', r));
    }
    ff.stdin.end();
    await new Promise((r, j) => ff.on('close', (c) => (c === 0 ? r() : j(new Error('ffmpeg ' + c)))));
    await page.evaluate(() => window.p2r.endCapture());
    log(`#${i + 1} captured ${frames} frames (${duration.toFixed(2)}s clip) in ${((Date.now() - t0) / 1000).toFixed(0)}s → ${path.basename(file)}`);
    results.push({ n: i + 1, prompt, file: path.basename(file), video: take.url, duration });
    fs.writeFileSync(path.join(OUT, 'takes.json'), JSON.stringify(results, null, 2));
  }
} finally {
  await browser.close();
}
log(`done: ${results.length}/${prompts.length} takes in ${OUT}`);
