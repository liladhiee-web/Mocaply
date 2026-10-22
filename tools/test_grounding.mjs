import test from "node:test";
import assert from "node:assert/strict";
import { planeAlignment, rotateVec, solveGrounding } from "../public/grounding.js";

const near = (a, b, eps, msg) => assert.ok(Math.abs(a - b) <= eps, `${msg ?? ""} ${a} != ${b} (eps ${eps})`);
const norm = (v) => {
  const l = Math.hypot(...v);
  return v.map((x) => x / l);
};
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

function basis(n) {
  const helper = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1];
  const u = norm(cross(n, helper));
  const v = cross(n, u);
  return [u, v];
}

test("planeAlignment is identity without a plane", () => {
  const a = planeAlignment(null);
  assert.deepEqual(a.q, [0, 0, 0, 1]);
  assert.equal(a.dy, 0);
});

test("planeAlignment maps a tilted plane onto y = 0", () => {
  const n = norm([0.14, 1, -0.07]);
  const d = 0.31;
  const { q, dy } = planeAlignment({ normal: n, d });
  assert.ok(Math.abs(dy - d) < 1e-9);
  const [u, v] = basis(n);
  const origin = n.map((c) => -d * c);
  for (const [a, b] of [[0, 0], [0.4, 0.2], [-0.3, 0.5], [0.1, -0.6]]) {
    const p = origin.map((c, i) => c + u[i] * a + v[i] * b);
    const out = rotateVec(q, p);
    near(out[1] + dy, 0, 1e-9, "plane point y");
  }
  const up = rotateVec(q, n);
  near(up[1], 1, 1e-9, "normal maps to up");
});

test("a small uniform float is pulled down to the floor", () => {
  const n = 60;
  const left = Array.from({ length: n }, () => [0.11, 0.05, 0]);
  const right = Array.from({ length: n }, () => [-0.11, 0.05, 0]);
  const g = solveGrounding({ left, right, floorY: 0, fps: 30 });
  near(0.05 + g.dy[30], 0, 0.002, "foot y after correction");
  assert.equal(g.bias, 0);
  assert.ok(g.contactRatio > 0.95, `contact ratio ${g.contactRatio}`);
});

test("a large uniform float is recovered with a bias", () => {
  const n = 60;
  const left = Array.from({ length: n }, () => [0.11, 0.2, 0]);
  const right = Array.from({ length: n }, () => [-0.11, 0.2, 0]);
  const g = solveGrounding({ left, right, floorY: 0, fps: 30 });
  near(0.2 + g.dy[30], 0, 0.002, "foot y after correction");
  near(g.bias, -0.2, 0.002, "bias");
});

test("slow float and sink within one clip are both corrected", () => {
  const n = 90;
  const left = [];
  const right = [];
  for (let i = 0; i < n; i++) {
    const y = 0.06 * Math.sin((i / (n - 1)) * Math.PI * 2);
    left.push([0.11, y, 0]);
    right.push([-0.11, y, 0]);
  }
  const g = solveGrounding({ left, right, floorY: 0, fps: 30 });
  for (let i = 6; i < n; i++) near(left[i][1] + g.dy[i], 0, 0.012, `frame ${i}`);
  for (let i = 0; i < n; i++) assert.ok(left[i][1] + g.dy[i] >= -1e-6, "never sinks");
});

test("a downward normal is treated as the same upward plane", () => {
  const n = norm([0.08, 1, -0.03]);
  const d = 0.12;
  const up = planeAlignment({ normal: n, d });
  const down = planeAlignment({ normal: n.map((c) => -c), d: -d });
  near(down.dy, up.dy, 1e-9, "offset sign preserved");
  const [u, v] = basis(n);
  const origin = n.map((c) => -d * c);
  for (const [a, b] of [[0, 0], [0.3, -0.2]]) {
    const p = origin.map((c, i) => c + u[i] * a + v[i] * b);
    const out = rotateVec(down.q, p);
    near(out[1] + down.dy, 0, 1e-9, "flipped plane point y");
  }
});

test("uniform sinking is lifted to the floor", () => {
  const n = 40;
  const left = Array.from({ length: n }, () => [0.11, -0.05, 0]);
  const right = Array.from({ length: n }, () => [-0.11, -0.05, 0]);
  const g = solveGrounding({ left, right, floorY: 0, fps: 30 });
  near(-0.05 + g.dy[20], 0, 0.002, "foot y after correction");
  for (let i = 0; i < n; i++) assert.ok(-0.05 + g.dy[i] >= -1e-6, "never sinks");
});

test("a planted foot stops sliding without touching the pose", () => {
  const n = 60;
  const left = [];
  const right = [];
  for (let i = 0; i < n; i++) {
    left.push([0.005 * i, 0, 0]); // 0.15 m/s drift while planted
    right.push([-0.18, 0.18, 0]); // swinging / lifted
  }
  const g = solveGrounding({ left, right, floorY: 0, fps: 30 });
  near(left[40][0] + g.dx[40], left[0][0], 0.002, "corrected planted x");
  near(left[59][0] + g.dx[59], left[0][0], 0.01, "corrected planted x at end");
  assert.ok(g.locked > 0.8, `locked frames ${g.locked}`);
});

test("in-place cycles keep their intended foot slide (lock disabled)", () => {
  const n = 60;
  const left = [];
  const right = [];
  for (let i = 0; i < n; i++) {
    left.push([0.005 * i, 0, 0]);
    right.push([-0.18, 0.18, 0]);
  }
  const g = solveGrounding({ left, right, floorY: 0, fps: 30, lock: false });
  assert.ok(g.dx.every((v) => v === 0), "no lateral x correction");
  assert.ok(g.dz.every((v) => v === 0), "no lateral z correction");
  assert.equal(g.locked, 0);
  near(left[40][0] + g.dx[40], left[40][0], 1e-9, "posed footprint untouched");
});

test("a jump keeps its arc and still lands on the floor", () => {
  const n = 50;
  const left = [];
  const right = [];
  for (let i = 0; i < n; i++) {
    const y = i < 10 || i >= 34 ? 0 : 0.55;
    left.push([0.1, y, 0]);
    right.push([-0.1, y, 0]);
  }
  const g = solveGrounding({ left, right, floorY: 0, fps: 30 });
  near(0.55 + g.dy[20], 0.55, 0.02, "flight height preserved");
  near(0 + g.dy[40], 0, 0.005, "landing grounded");
  for (let i = 0; i < n; i++) assert.ok(left[i][1] + g.dy[i] >= -1e-6, "never sinks");
});

test("missing foot data is a no-op", () => {
  const g = solveGrounding({ left: [], right: [], floorY: 0, fps: 30 });
  assert.equal(g.dx.length, 0);
  assert.equal(g.dy.length, 0);
  assert.equal(g.contactRatio, 0);
});

test("constant corrections are deterministic", () => {
  const n = 30;
  const left = Array.from({ length: n }, (_, i) => [0.1, 0.02 + 0.001 * Math.sin(i), 0]);
  const right = Array.from({ length: n }, () => [-0.1, 0.2, 0]);
  const a = solveGrounding({ left, right, floorY: 0, fps: 30 });
  const b = solveGrounding({ left, right, floorY: 0, fps: 30 });
  assert.deepEqual(a.dy, b.dy);
  assert.deepEqual(a.dx, b.dx);
});
