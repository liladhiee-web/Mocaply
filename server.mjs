// Mocaply — zero-dependency web server for AI motion capture and retargeting.
// Serves ./public (uploaded clips included) and proxies the local GVHMR worker.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC = path.join(ROOT, 'public');
const TAKES = path.join(ROOT, 'takes'); // exported takes for tools/render_blender.py
const UPLOADS = path.join(ROOT, 'public', 'uploads'); // clips loaded through the app

// --- .env -------------------------------------------------------------------
const envFile = path.join(ROOT, '.env');
if (fs.existsSync(envFile)) {
  for (const line of fs.readFileSync(envFile, 'utf8').split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/);
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2];
  }
}
const PORT = Number(process.env.PORT || 5177);
const HOST = process.env.HOST || '0.0.0.0';
const HQ_URL = process.env.HQ_URL || 'http://127.0.0.1:5178';

// --- request handling ----------------------------------------------------------
function readJson(req) {
  return new Promise((resolve, reject) => {
    let size = 0; const chunks = [];
    req.on('data', (c) => { size += c.length; if (size > 25e6) reject(new Error('body too large')); else chunks.push(c); });
    req.on('end', () => { try { resolve(JSON.parse(Buffer.concat(chunks).toString() || '{}')); } catch (e) { reject(e); } });
    req.on('error', reject);
  });
}

function send(res, status, body) {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(body));
}

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css',
  '.json': 'application/json', '.glb': 'model/gltf-binary', '.png': 'image/png', '.svg': 'image/svg+xml',
  '.mp4': 'video/mp4', '.webm': 'video/webm', '.mov': 'video/quicktime',
};

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  try {
    // Save an exported take (GLB body, ?name=…&source=<video url>) into ./takes for the Blender renderer.
    if (req.method === 'POST' && url.pathname === '/api/takes') {
      const name = (url.searchParams.get('name') || 'take').replace(/[^a-z0-9_-]+/gi, '_').slice(0, 80);
      const chunks = []; let size = 0;
      for await (const c of req) { size += c.length; if (size > 100e6) throw Object.assign(new Error('take too large'), { status: 413 }); chunks.push(c); }
      fs.mkdirSync(TAKES, { recursive: true });
      fs.writeFileSync(path.join(TAKES, name + '.glb'), Buffer.concat(chunks));
      const meta = { name, source: url.searchParams.get('source') || null, prompt: url.searchParams.get('prompt') || '', saved: new Date().toISOString() };
      fs.writeFileSync(path.join(TAKES, name + '.json'), JSON.stringify(meta, null, 2));
      console.log(`[take] saved takes/${name}.glb`);
      return send(res, 200, { path: `takes/${name}.glb` });
    }

    // Upload an existing video into public/uploads, where the
    // browser, the HQ worker (through /api/hq) and the Blender side-by-side can all reach it.
    if (req.method === 'POST' && url.pathname === '/api/upload') {
      const raw = (url.searchParams.get('name') || '').replace(/[^a-zA-Z0-9._-]+/g, '_');
      const ext = ['.mp4', '.webm', '.mov'].find((e) => raw.toLowerCase().endsWith(e)) || '.mp4';
      const stem = raw.slice(0, raw.length - ext.length).replace(/^\.+|\.+$/g, '').slice(0, 80) || 'clip';
      fs.mkdirSync(UPLOADS, { recursive: true });
      let name = stem + ext;
      for (let i = 2; fs.existsSync(path.join(UPLOADS, name)); i++) name = `${stem}_${i}${ext}`;
      const chunks = []; let size = 0;
      for await (const c of req) { size += c.length; if (size > 250e6) throw Object.assign(new Error('video too large (max ~250 MB)'), { status: 413 }); chunks.push(c); }
      if (size === 0) throw Object.assign(new Error('uploaded video is empty'), { status: 400 });
      fs.writeFileSync(path.join(UPLOADS, name), Buffer.concat(chunks));
      console.log(`[upload] saved public/uploads/${name} (${(size / 1e6).toFixed(1)} MB)`);
      return send(res, 200, { url: `/uploads/${name}`, bytes: size });
    }

    // HQ pass: forward to the local GVHMR worker (hq/worker.py).
    if (url.pathname === '/api/hq/health') {
      try {
        const r = await fetch(`${HQ_URL}/health`, { signal: AbortSignal.timeout(2000) });
        return send(res, 200, await r.json());
      } catch { return send(res, 200, { ok: false, error: 'worker offline' }); }
    }
    if (req.method === 'POST' && url.pathname === '/api/hq') {
      const b = await readJson(req);
      // The worker fetches the URL itself, so resolve app-relative upload URLs against this server.
      const video_url = b.video_url?.startsWith('/') ? `http://127.0.0.1:${PORT}${b.video_url}` : b.video_url;
      const r = await fetch(`${HQ_URL}/process`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_url, ...(b.hands ? { hands: b.hands } : {}) }),
      });
      const text = await r.text();
      res.writeHead(r.status, { 'Content-Type': 'application/json' });
      return res.end(text);
    }

    if (req.method === 'GET' || req.method === 'HEAD') {
      let p = decodeURIComponent(url.pathname);
      if (p === '/') p = '/index.html';
      const file = path.normalize(path.join(PUBLIC, p));
      if (!file.startsWith(PUBLIC) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
        res.writeHead(404); return res.end('not found');
      }
      const stat = fs.statSync(file);
      const ext = path.extname(file).toLowerCase();
      const contentType = MIME[ext] || 'application/octet-stream';
      const range = req.headers.range;

      if (range) {
        const parts = range.replace(/bytes=/, '').split('-');
        const start = parseInt(parts[0], 10);
        const end = parts[1] ? parseInt(parts[1], 10) : stat.size - 1;
        if (isNaN(start) || isNaN(end) || start >= stat.size || end >= stat.size || start > end) {
          res.writeHead(416, { 'Content-Range': `bytes */${stat.size}` });
          return res.end();
        }
        res.writeHead(206, {
          'Content-Range': `bytes ${start}-${end}/${stat.size}`,
          'Accept-Ranges': 'bytes',
          'Content-Length': end - start + 1,
          'Content-Type': contentType,
          'Cache-Control': 'no-cache',
        });
        if (req.method === 'HEAD') return res.end();
        return fs.createReadStream(file, { start, end }).pipe(res);
      }

      res.writeHead(200, {
        'Content-Type': contentType,
        'Content-Length': stat.size,
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'no-cache',
      });
      if (req.method === 'HEAD') return res.end();
      return fs.createReadStream(file).pipe(res);
    }

    res.writeHead(405); res.end();
  } catch (err) {
    console.error('[error]', err.message);
    send(res, err.status && err.status < 600 ? err.status : 500, { error: err.message });
  }
});

server.listen(PORT, HOST, () => console.log(`Mocaply → http://${HOST === '0.0.0.0' ? 'localhost' : HOST}:${PORT}`));
