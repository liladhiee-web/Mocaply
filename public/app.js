// Mocaply — load any clip → extract 3D motion via GVHMR neural network on GPU → Mixamo rig.
// Optimized for lag-free, rock-solid 60 FPS in-browser rendering.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { createStage } from './stage.js';
import { HAND_LIMITS, Rig, Track, trackToClip, blendSnapshots, smplGlobals, aaToQuat } from './retarget.js';

const $ = (s) => document.querySelector(s);
// Mixamo-rigged characters from three.js CDN
const THREE_MODELS = 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r169/examples/models/gltf/';
const CHARACTERS = {
  xbot: { name: 'X Bot', url: THREE_MODELS + 'Xbot.glb' },
  michelle: { name: 'Michelle', url: THREE_MODELS + 'Michelle.glb' },
  soldier: { name: 'Soldier', url: THREE_MODELS + 'Soldier.glb' },
};

// ---- settings ------------------------------------------------------------------
const DEFAULTS = {
  hq: true, rootMotion: true,
  character: 'xbot', swap: false, cinematic: false, hands: true,
};
const settings = (() => { try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem('p2r') || '{}') }; } catch { return { ...DEFAULTS }; } })();
const save = () => { try { localStorage.setItem('p2r', JSON.stringify(settings)); } catch {} };

function bindSetting(id, kind = 'value', onChange) {
  const el = $('#' + id);
  if (!el) return;
  if (kind === 'checked') el.checked = !!settings[id]; else el.value = settings[id];
  el.addEventListener('input', () => {
    settings[id] = kind === 'checked' ? el.checked : (el.type === 'range' ? Number(el.value) : el.value);
    save(); onChange?.(settings[id]);
  });
  onChange?.(settings[id]);
}

// ---- log / status -------------------------------------------------------------
function log(msg, cls = '') {
  const el = $('#log'); if (!el) return;
  const line = document.createElement('div');
  if (cls) line.className = cls;
  line.textContent = `${new Date().toLocaleTimeString([], { hour12: false })}  ${msg}`;
  el.appendChild(line); el.scrollTop = el.scrollHeight;
}
function pill(id, text, state) {
  const p = $('#pill-' + id);
  if (p) { p.textContent = text; p.className = 'pill ' + (state || ''); }
}
function banner(text) {
  const b = $('#banner');
  if (b) { b.textContent = text || ''; b.classList.toggle('on', !!text); }
}

async function api(path, body) {
  const res = await fetch(path, { method: body ? 'POST' : 'GET', headers: { 'Content-Type': 'application/json' }, body: body && JSON.stringify(body) });
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(json.error || `${path} → ${res.status}`);
  return json;
}

// ---- three.js stage -------------------------------------------------------------
const stage = createStage($('#stage'));
const { scene } = stage;
// Keep the characters centered in the part of the stage the video panel doesn't cover.
new ResizeObserver(() => { const v = $('#vidwrap'); if (v) stage.setInset(v.offsetLeft + v.offsetWidth); }).observe($('#vidwrap'));

// ---- character --------------------------------------------------------------------
let model = null, rig = null;

/** Load a Mixamo character: scaled to meters, facing +Z, centered with feet on y=0. */
async function loadModel(src, name) {
  const isFbx = /\.fbx$/i.test(name);
  const obj = await (isFbx ? new FBXLoader() : new GLTFLoader()).loadAsync(src);
  const root = isFbx ? obj : obj.scene;
  if (isFbx) root.scale.setScalar(0.01);         // Mixamo FBX is in centimeters
  root.updateMatrixWorld(true);
  // Retargeting assumes the character faces +Z (toward the camera); some exports face -Z.
  const armX = (side) => {
    let b; root.traverse((o) => { if (!b && o.isBone && o.name.replace(/^mixamorig\d*[:_]?/i, '') === side + 'Arm') b = o; });
    return b?.getWorldPosition(new THREE.Vector3()).x;
  };
  const lx = armX('Left'), rx = armX('Right');
  if (lx != null && rx != null && lx < rx) { root.rotation.y += Math.PI; root.updateMatrixWorld(true); }
  const box = new THREE.Box3().setFromObject(root);
  root.position.y -= box.min.y;
  root.position.x -= (box.min.x + box.max.x) / 2;
  root.position.z -= (box.min.z + box.max.z) / 2;
  root.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.frustumCulled = false; } });
  root.updateMatrixWorld(true);
  return root;
}

async function loadCharacter(src, name = 'X Bot') {
  pill('rig', 'loading ' + name.toLowerCase(), 'busy');
  const root = await loadModel(src, name);
  root.name = 'Character';
  const newRig = new Rig(root);
  if (model) { scene.remove(model); stage.removeContactShadow(model); }
  scene.add(root); stage.contactShadow(root); model = root; rig = newRig;
  // Re-bake HQ data for the new skeleton.
  for (const t of takes) { if (t.hqRaw) t.hq = bakeSmpl(t.hqRaw); }
  switchFrom = null;
  pill('rig', name.toLowerCase(), 'ok');
  log(`character: ${name} (${rig.order.length} bones, ${rig.height.toFixed(2)} m)`);
  resetCast();
}

// ---- character swap: one character at a time, the next one takes over on every loop -------------
const cast = {};        // key → { model, rig } for the non-driver characters, loaded on demand
let onStage = null;     // key of the character currently visible (null = the driver itself)

function resetCast() {
  for (const k of Object.keys(cast)) { scene.remove(cast[k].model); stage.removeContactShadow(cast[k].model); delete cast[k]; }
  onStage = null;
  if (model) model.visible = true;
}

async function swapCharacter() {
  const keys = Object.keys(CHARACTERS);
  const shown = onStage || settings.character;
  const next = keys[(keys.indexOf(shown) + 1) % keys.length];
  if (next === settings.character) { showOnStage(null); return; }
  if (!cast[next]) {
    const m = await loadModel(CHARACTERS[next].url, CHARACTERS[next].name);
    if (cast[next]) return;
    m.visible = false; scene.add(m);
    cast[next] = { model: m, rig: new Rig(m) };
  }
  showOnStage(next);
}

function showOnStage(key) {
  if (onStage && cast[onStage]) { cast[onStage].model.visible = false; stage.removeContactShadow(cast[onStage].model); }
  onStage = key;
  model.visible = !key;
  stage.removeContactShadow(model);
  if (key) { cast[key].model.visible = true; stage.contactShadow(cast[key].model); }
  else stage.contactShadow(model);
  pill('rig', (key ? CHARACTERS[key].name : (CHARACTERS[settings.character]?.name || 'character')).toLowerCase(), 'ok');
}

// ---- takes --------------------------------------------------------------------------
let takeSeq = 0;
const takes = [];
class Take {
  constructor(prompt) {
    this.id = ++takeSeq; this.prompt = prompt;
    this.videoUrl = null; this.status = 'ready';
    this.hq = null; this.hqRaw = null; this.hqStatus = 'off'; this.hqAt = 0;
    this.timings = {};
    this.ready = new Promise((r) => (this._resolveReady = r));
  }
  get best() { return this.hq; }
}

function renderTakes() {
  const el = $('#takes'); if (!el) return;
  el.innerHTML = '';
  for (const t of [...takes].reverse()) {
    const card = document.createElement('div');
    card.className = 'take' + (t === current ? ' playing' : '');
    const hq = {
      running: '<span class="pill busy"><span class="spin"></span> generating…</span>',
      done: `<span class="pill ok">hq ${t.timings.hq?.toFixed(1)}s</span>`,
      error: '<span class="pill bad">hq failed</span>',
    }[t.hqStatus] || '';
    card.innerHTML = `<div class="p" title="${escapeHtml(t.prompt)}">#${t.id} · ${escapeHtml(t.prompt)}</div>
      <div class="meta">${t.status === 'error' ? '<span class="pill bad">failed</span>' : ''}${hq}</div>
      <div class="actions"><button class="btn" data-a="play" ${t.videoUrl ? '' : 'disabled'}>▶ play</button>
      <button class="btn" data-a="glb" ${t.best ? '' : 'disabled'}>↓ glb</button></div>`;
    card.querySelector('[data-a=play]').onclick = () => playTake(t);
    card.querySelector('[data-a=glb]').onclick = () => exportTake(t);
    el.appendChild(card);
  }
}
const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// ---- HQ loading overlay -----------------------------------------------------------
const hqOverlay = $('#hq-overlay');
const hqTimer = $('#hq-timer');
let hqTimerInterval = null;

function showHqLoading(show) {
  if (!hqOverlay) return;
  if (show) {
    hqOverlay.style.display = 'flex';
    const t0 = performance.now();
    if (hqTimer) hqTimer.textContent = '00:00';
    clearInterval(hqTimerInterval);
    hqTimerInterval = setInterval(() => {
      const elapsed = Math.floor((performance.now() - t0) / 1000);
      const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
      const ss = String(elapsed % 60).padStart(2, '0');
      if (hqTimer) hqTimer.textContent = `${mm}:${ss}`;
    }, 250);
  } else {
    hqOverlay.style.display = 'none';
    clearInterval(hqTimerInterval);
  }
}

// ---- HQ (GVHMR) --------------------------------------------------------------------------
let hqOnline = false;
async function checkHQ() {
  try {
    const r = await api('/api/hq/health');
    hqOnline = !!r.ok;
    const isBusy = takes.some((t) => t.hqStatus === 'running');
    if (!isBusy) {
      pill('hq', r.ok ? `hq · ${(r.device || 'ready').replace(/^NVIDIA GeForce /, '').toLowerCase()}` : 'hq · ' + (r.error || 'not ready'), r.ok ? 'ok' : 'bad');
    }
  } catch {
    hqOnline = false;
    pill('hq', 'hq worker offline', 'bad');
  }
}

function runHQ(take) {
  const t0 = performance.now();
  take.hqT0 = t0;
  take.hqStatus = 'running';
  pill('hq', 'generating motion…', 'busy');
  if (current === take) {
    showHqLoading(true);
    video.pause();
  }
  renderTakes();

  take.hqPromise = api('/api/hq', { video_url: take.videoUrl, ...(take.handsMethod ? { hands: take.handsMethod } : {}) })
    .then((r) => {
      take.hqRaw = r;
      take.hq = bakeSmpl(r);
      take.hqAt = performance.now();
      take.hqStatus = 'done';
      take.timings.hq = (performance.now() - t0) / 1000;
      log(`#${take.id} hq motion generated in ${take.timings.hq.toFixed(1)}s (${r.n} frames, worker ${r.elapsed?.toFixed(1)}s)`, 'hq');
      pill('hq', `hq · ${take.timings.hq.toFixed(1)}s ready`, 'ok');
      if (current === take) {
        showHqLoading(false);
        video.currentTime = 0;
        video.play().catch(() => {});
      }
      banner(`✓ High-Quality GVHMR motion ready (${take.timings.hq.toFixed(1)}s)! Playing on rig.`);
      setTimeout(() => banner(''), 4500);
    })
    .catch((e) => {
      take.hqStatus = 'error';
      if (current === take) showHqLoading(false);
      log(`#${take.id} hq generation failed: ${e.message}`, 'err');
      pill('hq', 'hq failed', 'bad');
      banner(`HQ Generation failed: ${e.message}`);
      setTimeout(() => banner(''), 5000);
    })
    .finally(() => {
      renderTakes();
    });
}

// Finger tracking from the HQ worker: per hand, 21 wrist-relative keypoints per frame (GVHMR world axes).
const _kp = Array.from({ length: 21 }, () => new THREE.Vector3());
function applyHandsFrame(hands, f, yawFix) {
  // WiLoR's MANO hands are anatomically valid already; joint limits only help MediaPipe's noisy depth.
  HAND_LIMITS.enabled = hands?.method !== 'wilor';
  for (const [src, side] of [['R', 'Right'], ['L', 'Left']]) {
    const h = hands?.[src];
    if (settings.hands !== false && h?.valid[f]) {
      const o = f * 63;
      for (let j = 0; j < 21; j++) _kp[j].set(h.kp[o + j * 3], h.kp[o + j * 3 + 1], h.kp[o + j * 3 + 2]).applyQuaternion(yawFix);
      rig.applyHand(side, _kp);
    } else rig.relaxHand(side);
  }
}

/** GVHMR SMPL params → Track of Mixamo local rotations + grounded hips path. */
function bakeSmpl(r) {
  const { n, fps, global_orient: GO, body_pose: BP, transl: T, leg_length } = r;
  // Heading: turn GVHMR's world so the character faces the viewer the way the person faces the camera.
  const fWorld = new THREE.Vector3(0, 0, 1).applyQuaternion(aaToQuat(GO[0], GO[1], GO[2]));
  let targetYaw = 0;
  if (r.incam_global_orient0) {
    const [a, b, c] = r.incam_global_orient0;
    const fc = new THREE.Vector3(0, 0, 1).applyQuaternion(aaToQuat(a, b, c));
    targetYaw = Math.atan2(fc.x, -fc.z);
  }
  const yawFix = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), targetYaw - Math.atan2(fWorld.x, fWorld.z));
  const scale = rig.legLength / (leg_length || 0.85);
  const track = new Track(rig.order.length);
  const lows = [];
  const d = new THREE.Vector3();
  for (let f = 0; f < n; f++) {
    rig.reset();
    rig.applySmplGlobal(smplGlobals(GO.subarray ? GO.subarray(f * 3, f * 3 + 3) : GO.slice(f * 3, f * 3 + 3), BP.slice(f * 63, f * 63 + 63), yawFix));
    if (settings.rootMotion) {
      d.set(T[f * 3] - T[0], T[f * 3 + 1] - T[1], T[f * 3 + 2] - T[2]).applyQuaternion(yawFix).multiplyScalar(scale);
      const p = rig.rest.Hips.wp.clone().add(d);
      rig.setHipsWorld(p);
    }
    applyHandsFrame(r.hands, f, yawFix);
    lows.push(rig.lowestFoot());
    track.push(f / fps, rig.snapshot());
  }
  // Ground: the typical lowest foot sits on the floor; never sink below it.
  const sorted = [...lows].sort((a, b) => a - b);
  const offset = rig.floorY - sorted[Math.floor(sorted.length * 0.08)];
  track.p.forEach((p, f) => { p[1] += offset + Math.max(0, rig.floorY - (lows[f] + offset)); });
  rig.reset();
  return track;
}

// ---- playback -----------------------------------------------------------------------------
const video = $('#video');
let current = null;
const queue = [];
let lastRenderedHips = new THREE.Vector3();
let switchFrom = null, switchAt = 0;

function enqueue(take) {
  queue.push(take);
  if (capturing) return;
  if (!current) playNext();
}

function playNext() {
  const t = queue.shift();
  if (t) playTake(t);
}

function playTake(take) {
  if (!take || !take.videoUrl) return;
  if (current && rig) { switchFrom = rig.snapshot(); switchAt = performance.now(); }
  take.started = true;
  current = take;
  const qi = queue.indexOf(take); if (qi >= 0) queue.splice(qi, 1);

  const vidEmpty = $('#vid-empty');
  if (vidEmpty) vidEmpty.style.display = 'none';
  video.style.display = 'block';
  if (take.videoUrl.startsWith('blob:')) {
    video.removeAttribute('crossorigin');
  } else {
    video.crossOrigin = 'anonymous';
  }
  video.src = take.videoUrl;
  video.onloadedmetadata = () => $('#vidwrap')?.classList.toggle('portrait', video.videoHeight > video.videoWidth);

  if (take.hqStatus === 'running') {
    showHqLoading(true);
    video.currentTime = 0;
    video.pause();
  } else {
    showHqLoading(false);
    video.currentTime = 0;
    video.play().catch(() => {});
  }
  renderTakes();
}

video.addEventListener('ended', () => {
  if (settings.swap && rig) swapCharacter().catch((e) => log('swap failed: ' + e.message, 'err'));
  if (queue.length) playNext();
  else { video.currentTime = 0; video.play().catch(() => {}); }
});

// ---- render loop (smooth 60fps) -------------------------------------------------------------
const bufA = { q: null, p: new Float32Array(3) }, bufB = { q: null, p: new Float32Array(3) }, bufC = { q: null, p: new Float32Array(3) };

let lastRaf = 0;
let capturing = false;
function frame() {
  requestAnimationFrame(frame);
  lastRaf = performance.now();
  if (!capturing) tick();
}
setInterval(() => { if (!capturing && performance.now() - lastRaf > 100) tick(); }, 33);

function tick() {
  if (rig && current) {
    const n = rig.order.length * 4;
    for (const b of [bufA, bufB, bufC]) if (!b.q || b.q.length !== n) b.q = new Float32Array(n);
    const t = video.currentTime;
    let snap = current.hq ? current.hq.sample(t, bufB) : null;
    let src = current.hq ? 'hq · gvhmr' : current.hqStatus === 'running' ? 'generating motion…' : '—';

    if (snap && switchFrom) {
      const w = Math.min(1, (performance.now() - switchAt) / 250);
      if (w < 1) snap = blendSnapshots(switchFrom.q, switchFrom.p, snap.q, snap.p, w, snap === bufC ? bufA : bufC);
      else switchFrom = null;
    }
    if (snap) {
      rig.applySnapshot(snap);
      rig.hipsWorld(lastRenderedHips);
    }
    const hud = $('#hud-src');
    if (hud) {
      hud.textContent = src;
      hud.className = 'tag ' + (current.hq ? 'src-hq' : '');
    }
    const hudTime = $('#hud-time');
    if (hudTime) {
      hudTime.textContent = `#${current.id}  ${t.toFixed(2)}s`;
    }

    const hudHq = $('#hud-hq');
    const vidBadge = $('#vid-badge');
    if (current.hqStatus === 'running') {
      const elapsed = ((performance.now() - (current.hqT0 || performance.now())) / 1000).toFixed(1);
      if (hudHq) {
        hudHq.style.display = 'block';
        hudHq.className = 'tag hq-busy';
        hudHq.innerHTML = `<span class="spin"></span> generating 3d motion · ${elapsed}s`;
      }
      if (vidBadge) {
        vidBadge.style.display = 'flex';
        vidBadge.className = 'vid-badge busy';
        vidBadge.innerHTML = `<span class="spin"></span> Generating 3D Motion (GPU) · ${elapsed}s`;
      }
    } else if (current.hq && (performance.now() - current.hqAt) < 6000) {
      if (hudHq) {
        hudHq.style.display = 'block';
        hudHq.className = 'tag hq-done';
        hudHq.textContent = `✓ hq motion active (${current.timings.hq?.toFixed(1) || ''}s)`;
      }
      if (vidBadge) {
        vidBadge.style.display = 'flex';
        vidBadge.className = 'vid-badge done';
        vidBadge.innerHTML = `✓ HQ Motion Active`;
      }
    } else {
      if (hudHq) hudHq.style.display = 'none';
      if (vidBadge) vidBadge.style.display = 'none';
    }
  }
  if (rig) {
    const shown = onStage ? cast[onStage].rig : rig;
    if (shown !== rig) shown.copyPoseFrom(rig);
    stage.placeContactShadow(shown.model, shown.hipsWorld(), shown.floorY, shown.height);
  }
  stage.update(rig ? lastRenderedHips : null);
}

// ---- export --------------------------------------------------------------------------------
async function exportTake(take) {
  const track = take.best;
  if (!track) return;
  const name = `take${take.id}_${take.prompt.replace(/[^a-z0-9]+/gi, '_').slice(0, 40)}_hq`;
  const clip = trackToClip(rig, track, name);
  const glb = await new GLTFExporter().parseAsync(model, { binary: true, animations: [clip] });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([glb], { type: 'model/gltf-binary' }));
  a.download = name + '.glb'; a.click();
  const source = take.videoUrl?.startsWith('http') ? take.videoUrl
    : take.videoUrl?.startsWith('/uploads/') ? 'public' + take.videoUrl : null;
  const q = new URLSearchParams({ name, prompt: take.prompt, ...(source ? { source } : {}) });
  fetch('/api/takes?' + q, { method: 'POST', body: glb }).then((r) => r.ok && log(`saved takes/${name}.glb`, 'ok')).catch(() => {});
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  log(`exported ${name}.glb`, 'ok');
}

// ---- UI wiring --------------------------------------------------------------------------------
bindSetting('swap', 'checked', (on) => { if (!on && model) showOnStage(null); });
bindSetting('cinematic', 'checked', (on) => stage.setCinematic(on));
bindSetting('character', 'value', (c) => { if (rig && CHARACTERS[c]) loadCharacter(CHARACTERS[c].url, CHARACTERS[c].name).catch((e) => log('character failed: ' + e.message, 'err')); });
bindSetting('hq', 'checked');
bindSetting('hands', 'checked', () => { if (rig) for (const t of takes) if (t.hqRaw) t.hq = bakeSmpl(t.hqRaw); });
bindSetting('rootMotion', 'checked', () => { if (rig) for (const t of takes) if (t.hqRaw) t.hq = bakeSmpl(t.hqRaw); });

$('#btn-char').onclick = () => $('#file-char').click();
$('#file-char').onchange = (e) => {
  const f = e.target.files[0]; if (!f) return;
  const url = URL.createObjectURL(f);
  loadCharacter(url, f.name).catch((err) => { log('character failed: ' + err.message, 'err'); pill('rig', 'character failed', 'bad'); });
};

async function handleVideoFile(f) {
  banner(`loading ${f.name}…`);
  const take = new Take(`(file) ${f.name}`);
  let uploaded = false;
  try {
    const res = await fetch(`/api/upload?name=${encodeURIComponent(f.name)}`, { method: 'POST', body: f });
    const j = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(j.error || `upload → ${res.status}`);
    take.videoUrl = j.url;
    take.status = 'ready';
    uploaded = true;
    log(`#${take.id} clip uploaded (${(j.bytes / 1e6).toFixed(1)} MB)`);
  } catch (err) {
    take.videoUrl = URL.createObjectURL(f);
    take.status = 'ready';
    log(`upload failed (${err.message}) — playing local file`, 'err');
  }
  takes.push(take);
  renderTakes();
  playTake(take);

  if (uploaded) {
    if (hqOnline) {
      banner(`Extracting 3D motion with GVHMR neural network on GPU…`);
      runHQ(take);
    } else {
      await checkHQ();
      if (hqOnline) {
        banner(`Extracting 3D motion with GVHMR neural network on GPU…`);
        runHQ(take);
      } else {
        banner(`HQ Worker offline. Run Google Colab or tools/colab_runner.py to generate 3D motion.`);
        log(`HQ Worker is offline! Connect Google Colab or your GPU runner to extract 3D motion.`, 'err');
      }
    }
  } else {
    banner(`Could not upload clip to server for HQ processing.`);
  }
}

$('#btn-vid').onclick = () => $('#file-vid').click();
$('#file-vid').onchange = (e) => {
  const f = e.target.files[0]; e.target.value = ''; if (!f) return;
  handleVideoFile(f);
};

window.addEventListener('dragover', (e) => {
  e.preventDefault();
  if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
});
window.addEventListener('drop', (e) => {
  e.preventDefault();
  const f = e.dataTransfer?.files?.[0];
  if (!f) return;
  if (/\.(mp4|webm|mov)$/i.test(f.name) || f.type.startsWith('video/')) {
    handleVideoFile(f);
  } else if (/\.(glb|gltf|fbx)$/i.test(f.name)) {
    const url = URL.createObjectURL(f);
    loadCharacter(url, f.name).catch((err) => { log('character failed: ' + err.message, 'err'); pill('rig', 'character failed', 'bad'); });
  }
});

// ---- capture zoom -------------------------------------------------------------
let zoomBase = null, zoomTgt = null;
const smootherstep = (x) => x * x * x * (x * (x * 6 - 15) + 10);
function applyCaptureZoom(k, bone, t = 0) {
  const r = onStage ? cast[onStage].rig : rig;
  if (k <= 0.0005) {
    if (zoomBase) { stage.setFocusLock(null); stage.camera.position.copy(zoomBase.pos); stage.controls.target.copy(zoomBase.tgt); zoomBase = null; }
    video.style.transform = '';
    zoomTgt = null;
    return;
  }
  if (!zoomBase) {
    const off = stage.camera.position.clone().sub(stage.controls.target);
    zoomBase = { pos: stage.camera.position.clone(), tgt: stage.controls.target.clone(), dir: off.clone().normalize(), dist: off.length() };
  }
  const e = smootherstep(Math.min(1, Math.max(0, k)));
  const hand = r.bones[bone].getWorldPosition(new THREE.Vector3());
  const mid = r.bones[bone.replace('Hand', 'HandMiddle2')];
  const want = mid ? hand.lerp(mid.getWorldPosition(new THREE.Vector3()), 0.6) : hand;
  zoomTgt = zoomTgt ? zoomTgt.lerp(want, 0.3) : want.clone();
  const tgt = zoomBase.tgt.clone().lerp(zoomTgt, e);
  const closeDir = new THREE.Vector3(bone.startsWith('Right') ? 0.06 : -0.06, 0.04, 1).normalize();
  const qArc = new THREE.Quaternion().setFromUnitVectors(zoomBase.dir, closeDir);
  const dir = zoomBase.dir.clone().applyQuaternion(new THREE.Quaternion().slerp(qArc, e));
  dir.applyAxisAngle(new THREE.Vector3(0, 1, 0), 0.09 * Math.sin(t * 0.7) * e);
  const dist = Math.exp(Math.log(zoomBase.dist) + (Math.log(0.52) - Math.log(zoomBase.dist)) * e);
  stage.setFocusLock(tgt);
  stage.camera.position.copy(tgt).addScaledVector(dir, dist);
  stage.update(null);
  const scale = Math.exp(Math.log(3.2) * e);
  video.style.transformOrigin = `50% 35%`;
  video.style.transform = `scale(${scale})`;
}

// Debug handle for console
window.p2r = {
  get current() { return current; }, get rig() { return rig; }, takes, video,
  stage, HAND_LIMITS, rebake() { for (const t of takes) if (t.hqRaw) t.hq = bakeSmpl(t.hqRaw); },
  focusBone(name, d = 0.6, dir = [0.15, 0.1, 1]) {
    const r = onStage ? cast[onStage].rig : rig;
    const p = r.bones[name].getWorldPosition(new THREE.Vector3());
    stage.setCinematic(false); stage.controls.target.copy(p);
    stage.camera.position.copy(p).add(new THREE.Vector3(...dir).normalize().multiplyScalar(d));
    stage.setFocusLock?.(p);
  },
  take(id) { const t = takes.find((x) => x.id === id); return t && { id: t.id, status: t.status, hq: t.hqStatus, url: t.videoUrl, duration: t.hq?.duration ?? null }; },
  async beginCapture(id) {
    const t = takes.find((x) => x.id === id);
    if (!t?.hq) throw new Error('take has no HQ motion');
    capturing = true; video.loop = false;
    if (current !== t) playTake(t);
    video.pause();
    await new Promise((r) => (video.readyState >= 2 ? r() : video.addEventListener('loadeddata', r, { once: true })));
    renderTakes();
    return { duration: video.duration, frames: Math.floor(video.duration * 30) };
  },
  async renderAt(t, zoom = null) {
    video.pause();
    if (Math.abs(video.currentTime - t) > 1e-4) {
      const seeked = new Promise((r) => video.addEventListener('seeked', r, { once: true }));
      video.currentTime = t; await seeked;
    }
    tick();
    if (zoom) applyCaptureZoom(zoom.k, zoom.bone || 'RightHand', zoom.t || 0);
    await Promise.race([new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))), new Promise((r) => setTimeout(r, 80))]);
  },
  endCapture() { capturing = false; video.play().catch(() => {}); },
  playUrl(u, hands, prompt) { const t = new Take(prompt || 'debug: ' + u.slice(-24)); t.videoUrl = u; t.status = 'ready'; t.handsMethod = hands; takes.push(t); if (hqOnline) runHQ(t); enqueue(t); return { id: t.id }; },
};

// ---- boot --------------------------------------------------------------------------------------
frame();
const startChar = CHARACTERS[settings.character] || CHARACTERS.xbot;
loadCharacter(startChar.url, startChar.name).catch((e) => { log('character failed: ' + e.message, 'err'); pill('rig', 'character failed', 'bad'); });
checkHQ(); setInterval(checkHQ, 10000);
log('ready. load a clip below to begin');
