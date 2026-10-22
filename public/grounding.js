// Ground-plane alignment and foot-contact solving.
//
// The HQ worker reconstructs the floor from the SMPL-X soles (hq/ground.py) and
// returns its plane in GVHMR world space. This module is pure math (no three.js)
// so the behaviour is unit-testable:
//
//   • planeAlignment()  builds the rotation that puts the detected floor on y=0
//   • solveGrounding()  turns per-frame foot soles into root corrections:
//                       vertical contact, no sinking, no floating, and a
//                       horizontal lock that removes foot sliding while planted.
//
// Only the root/hips are corrected. Bone rotations are never touched, so the
// recovered pose and motion stay intact.

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const yOf = (p) => (Array.isArray(p) ? p[1] : p.y);
const xOf = (p) => (Array.isArray(p) ? p[0] : p.x);
const zOf = (p) => (Array.isArray(p) ? p[2] : p.z);

export const GROUND_DEFAULTS = {
  // Contact detection (metres, metres/second). A planted foot can hover a few
  // centimetres from reconstruction error, so height alone is deliberately
  // generous; speed and a minimum contact run reject real steps and blips.
  startHeight: 0.12,
  releaseHeight: 0.18,
  startSpeed: 0.5,
  releaseSpeed: 0.8,
  minContactRun: 3,
  // Lateral drift is corrected up to this many metres per contact run.
  maxLock: 0.35,
  lockGain: 0.8,
  // Fraction of the clip assumed to be in ground contact (used to recover a
  // constant vertical bias before contact detection).
  biasQuantile: 0.1,
};

export function rotateVec(q, v) {
  const [x, y, z, w] = q;
  const [vx, vy, vz] = v;
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [vx + w * tx + y * tz - z * ty, vy + w * ty + z * tx - x * tz, vz + w * tz + x * ty - y * tx];
}

/** Rotation taking the plane normal to +Y, plus the Y shift that puts the plane on y = 0. */
export function planeAlignment(plane) {
  if (!plane || !Array.isArray(plane.normal) || plane.normal.length !== 3) {
    return { q: [0, 0, 0, 1], dy: 0, tiltDeg: 0 };
  }
  let [x, y, z] = plane.normal.map(Number);
  const len = Math.hypot(x, y, z);
  if (!(len > 1e-8) || !Number.isFinite(len)) return { q: [0, 0, 0, 1], dy: 0, tiltDeg: 0 };
  x /= len;
  y /= len;
  z /= len;
  let d = Number.isFinite(plane.d) ? Number(plane.d) : 0;
  // Represent the same plane with an upward normal; a downward normal would
  // otherwise rotate the character upside down.
  if (y < 0) {
    x = -x;
    y = -y;
    z = -z;
    d = -d;
  }
  const dot = clamp(y, -1, 1);
  const tiltDeg = (Math.acos(dot) * 180) / Math.PI;
  if (dot > 1 - 1e-9) return { q: [0, 0, 0, 1], dy: d, tiltDeg };
  // cross(normal, up) = (-z, 0, x); q = [axis*sin(θ/2), cos(θ/2)].
  const s = Math.sqrt(Math.max(0, 2 * (1 + dot)));
  if (!(s > 1e-9)) {
    // Antiparallel: 180° about X.
    return { q: [1, 0, 0, 0], dy: d, tiltDeg };
  }
  const q = [-z / s, 0, x / s, s / 2];
  const ql = Math.hypot(q[0], q[1], q[2], q[3]);
  return { q: q.map((v) => v / ql), dy: d, tiltDeg };
}

function quantile(values, q) {
  const a = values.filter(Number.isFinite).sort((p, r) => p - r);
  if (!a.length) return 0;
  const i = clamp(q, 0, 1) * (a.length - 1);
  const lo = Math.floor(i);
  const hi = Math.min(a.length - 1, lo + 1);
  return a[lo] + (a[hi] - a[lo]) * (i - lo);
}

function footSpeed(points, fps, i, fallback) {
  const a = points[i];
  const b = points[i > 0 ? i - 1 : Math.min(1, points.length - 1)];
  if (!a || !b) return fallback;
  return Math.hypot(xOf(a) - xOf(b), zOf(a) - zOf(b)) * fps;
}

function smooth3(values) {
  return values.map((v, i) => {
    const a = values[Math.max(0, i - 1)];
    const b = values[Math.min(values.length - 1, i + 1)];
    return (a + v + b) / 3;
  });
}

/**
 * Solve root corrections from per-frame foot sole positions.
 *
 * `left` / `right` are arrays (one entry per frame) of a foot sole's world
 * position `[x, y, z]` *after* ground-plane alignment. `floorY` is the
 * character's floor (normally 0). `lock` disables the horizontal anti-slide
 * correction (used when root motion is switched off, where feet are meant to
 * slide through an in-place cycle).
 *
 * Returns `{dx, dz, dy, contact, contactRatio, locked, bias}` where dx/dz/dy
 * are per-frame root translations to add to the hips world position.
 */
export function solveGrounding({ left, right, floorY = 0, fps = 30, lock = true, opts = {} }) {
  const o = { ...GROUND_DEFAULTS, ...opts };
  const n = Math.min(left?.length || 0, right?.length || 0);
  const zero = { dx: [], dz: [], dy: [], contact: new Uint8Array(0), contactRatio: 0, locked: 0, bias: 0 };
  if (!n) return zero;

  const fpsSafe = Number.isFinite(fps) && fps > 1 ? fps : 30;
  const yL = left.map(yOf);
  const yR = right.map(yOf);
  const lows = yL.map((y, i) => Math.min(y, yR[i]));
  if (!lows.every(Number.isFinite)) return zero;

  const horizontalSpeed = (points) => smooth3(points.map((_, i) => footSpeed(points, fpsSafe, i, 0)));
  const verticalSpeed = (ys) => smooth3(ys.map((y, i) => (i ? Math.abs(y - ys[i - 1]) * fpsSafe : 0)));
  const spdL = horizontalSpeed(left);
  const spdR = horizontalSpeed(right);
  const vertL = verticalSpeed(yL);
  const vertR = verticalSpeed(yR);

  const countContacts = (c) => {
    let k = 0;
    for (let i = 0; i < n; i++) if (c[i]) k++;
    return k;
  };

  const detect = (offset) => {
    const flags = new Uint8Array(n);
    let activeL = false;
    let activeR = false;
    for (let i = 0; i < n; i++) {
      const hL = yL[i] + offset - floorY;
      const hR = yR[i] + offset - floorY;
      const sL = Math.hypot(spdL[i], vertL[i]);
      const sR = Math.hypot(spdR[i], vertR[i]);
      const startL = hL < o.startHeight && sL < o.startSpeed;
      const stayL = hL < o.releaseHeight && sL < o.releaseSpeed;
      const startR = hR < o.startHeight && sR < o.startSpeed;
      const stayR = hR < o.releaseHeight && sR < o.releaseSpeed;
      activeL = activeL ? stayL : startL;
      activeR = activeR ? stayR : startR;
      flags[i] = (activeL ? 1 : 0) | (activeR ? 2 : 0);
    }
    // Drop implausibly short runs (the apex of a slow foot lift, a jitter blip).
    for (const bit of [1, 2]) {
      let i = 0;
      while (i < n) {
        if (!(flags[i] & bit)) {
          i++;
          continue;
        }
        let j = i;
        while (j < n && (flags[j] & bit)) j++;
        if (j - i < o.minContactRun) for (let k = i; k < j; k++) flags[k] &= ~bit;
        i = j;
      }
    }
    return flags;
  };

  // A clip that floats uniformly by more than the contact threshold has no
  // contact frames at all; recover its offset from the low decile once, then
  // detect against that height. Normal clips keep a zero bias and are handled
  // frame by frame.
  let bias = 0;
  let contact = detect(0);
  if (countContacts(contact) < Math.max(2, Math.ceil(n * 0.03))) {
    bias = clamp(floorY - quantile(lows, o.biasQuantile), -1, 1);
    contact = detect(bias);
  }

  // Raw vertical target: planted foot on the floor; airborne frames may only be
  // lifted (never pulled down), so jumps keep their arc.
  const rawDy = new Array(n);
  for (let i = 0; i < n; i++) {
    if (contact[i]) {
      const ys = [];
      if (contact[i] & 1) ys.push(yL[i]);
      if (contact[i] & 2) ys.push(yR[i]);
      rawDy[i] = floorY - Math.min(...ys);
    } else {
      rawDy[i] = Math.max(0, floorY - lows[i]);
    }
  }

  // Smoothing: follow contact targets closely and ease through airborne frames
  // so take-off/landing do not pop. The hard clamp below forbids penetration.
  const dy = new Array(n);
  let sm = rawDy[0];
  for (let i = 0; i < n; i++) {
    const d = rawDy[i];
    const alpha = contact[i] ? 0.85 : 0.4;
    sm = Math.abs(d - sm) > 0.12 ? d : sm + (d - sm) * alpha;
    dy[i] = Math.max(sm, floorY - lows[i]);
  }

  // Horizontal foot lock. The anchor is fixed when a foot plants, and the
  // correction persists after it releases so the body never snaps back. Only
  // active with root motion: in-place cycles are supposed to slide.
  const dx = new Array(n).fill(0);
  const dz = new Array(n).fill(0);
  let locked = 0;
  if (lock) {
    const anchor = { 1: null, 2: null };
    let offX = 0;
    let offZ = 0;
    for (let i = 0; i < n; i++) {
      const wants = [];
      for (const bit of [1, 2]) {
        const active = contact[i] & bit;
        if (!active) {
          anchor[bit] = null;
          continue;
        }
        const p = bit === 1 ? left[i] : right[i];
        if (!anchor[bit]) anchor[bit] = { x: xOf(p) + offX, z: zOf(p) + offZ };
        wants.push({ x: anchor[bit].x - xOf(p), z: anchor[bit].z - zOf(p) });
      }
      if (wants.length) {
        const wantX = clamp(wants.reduce((s, w) => s + w.x, 0) / wants.length, -o.maxLock, o.maxLock);
        const wantZ = clamp(wants.reduce((s, w) => s + w.z, 0) / wants.length, -o.maxLock, o.maxLock);
        offX = clamp(offX + (wantX - offX) * o.lockGain, -o.maxLock, o.maxLock);
        offZ = clamp(offZ + (wantZ - offZ) * o.lockGain, -o.maxLock, o.maxLock);
        locked++;
      }
      dx[i] = offX;
      dz[i] = offZ;
    }
  }

  let contacts = 0;
  for (let i = 0; i < n; i++) if (contact[i]) contacts++;
  return { dx, dz, dy, contact, contactRatio: contacts / n, locked: locked / n, bias };
}
