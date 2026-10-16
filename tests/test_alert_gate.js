// Tests for static/alerts.js (AlertGate). Needs Node.js:  node tests/test_alert_gate.js
const assert = require("assert");
const path = require("path");
const { AlertGate, PATTERNS, leadingSilence } = require(path.join(__dirname, "..", "static", "alerts.js"));

let passed = 0, failed = 0;
function test(name, fn) {
  try { fn(); passed++; console.log("  PASS  " + name); }
  catch (e) { failed++; console.log("  FAIL  " + name + "\n        " + e.message); }
}
// run a list of kinds as consecutive frames, 160 ms apart; returns the list of sounds [kind, frameIndex]
function run(gate, frames, startMs = 0, stepMs = 160) {
  const sounds = [];
  frames.forEach((k, i) => { const r = gate.feed(k, startMs + i * stepMs); if (r) sounds.push([r, i]); });
  return sounds;
}
const rep = (k, n) => Array(n).fill(k);

console.log("\nAlertGate duplicate-alert prevention");

test("a jammer alert sounds exactly once while the jammer stays active (100 frames)", () => {
  const s = run(new AlertGate(), [...rep(null, 5), ...rep("jammer", 100)]);
  assert.deepStrictEqual(s.map(x => x[0]), ["jammer"]);
});
test("it sounds only after 3 confirmed frames, not on the first one", () => {
  const s = run(new AlertGate(), [null, null, "jammer", "jammer", "jammer", "jammer"]);
  assert.strictEqual(s.length, 1); assert.strictEqual(s[0][1], 4);          // frame index 4 = third jammer frame
});
test("a single flickering frame never sounds", () => {
  const s = run(new AlertGate(), [null, null, "jammer", null, null, null, "jammer", null, null]);
  assert.strictEqual(s.length, 0);
});
test("jammer -> normal -> jammer inside the cooldown does not sound twice", () => {
  const g = new AlertGate({ cooldownMs: 8000 });
  const s = run(g, [...rep("jammer", 5), ...rep(null, 5), ...rep("jammer", 5)]);      // 15 frames = 2.4 s
  assert.strictEqual(s.length, 1);
});
test("after the cooldown a new jammer event sounds again", () => {
  const g = new AlertGate({ cooldownMs: 8000 });
  const s = [...run(g, [...rep("jammer", 5), ...rep(null, 5)], 0), ...run(g, rep("jammer", 5), 20000)];
  assert.deepStrictEqual(s.map(x => x[0]), ["jammer", "jammer"]);
});
test("the alert re-arms only after the state has cleared (needs 3 clear frames)", () => {
  const g = new AlertGate({ cooldownMs: 0 });
  assert.strictEqual(run(g, [...rep("jammer", 4), null, null, ...rep("jammer", 4)]).length, 1);   // clear for only 2 frames
  const g2 = new AlertGate({ cooldownMs: 0 });
  assert.strictEqual(run(g2, [...rep("jammer", 4), ...rep(null, 3), ...rep("jammer", 4)]).length, 2);
});
test("jammer -> unknown (a different kind) sounds a second, different alert", () => {
  const s = run(new AlertGate({ cooldownMs: 0, holdMs: 0 }), [...rep("jammer", 4), ...rep("unknown", 4)]);
  assert.deepStrictEqual(s.map(x => x[0]), ["jammer", "unknown"]);
});
test("a lower-priority alert is silent right after a jammer alert (link status lags behind)", () => {
  const s = run(new AlertGate(), [...rep("jammer", 4), ...rep(null, 3), ...rep("degraded", 4)]);   // ~1.8 s later
  assert.deepStrictEqual(s.map(x => x[0]), ["jammer"]);
});
test("a degraded alert alone does sound", () => {
  assert.deepStrictEqual(run(new AlertGate(), rep("degraded", 6)).map(x => x[0]), ["degraded"]);
});
test("a silent (consumed) event is not replayed later: state stays the same", () => {
  const g = new AlertGate();
  const s = run(g, [...rep("jammer", 4), ...rep("degraded", 50)]);              // degraded blocked by priority hold, then still active
  assert.deepStrictEqual(s.map(x => x[0]), ["jammer"]);
});
test("reset() (power off) clears everything so a new session can alert", () => {
  const g = new AlertGate();
  assert.strictEqual(run(g, rep("jammer", 5)).length, 1);
  g.reset();
  assert.strictEqual(run(g, rep("jammer", 5), 500).length, 1);
});
test("normal operation (no alert states) never sounds", () => {
  assert.strictEqual(run(new AlertGate(), rep(null, 500)).length, 0);
});
test("per-kind confirmation: a 3-frame UNKNOWN (classifier settling between two signals) is silent, a real one sounds", () => {
  const cf = { jammer: 3, unknown: 6, degraded: 5, clear: 3 };
  assert.strictEqual(run(new AlertGate({ confirmFrames: cf }), [...rep(null, 4), ...rep("unknown", 3), ...rep("jammer", 6)]).map(x => x[0]).join(), "jammer");
  assert.strictEqual(run(new AlertGate({ confirmFrames: cf }), [...rep(null, 4), ...rep("unknown", 8)]).map(x => x[0]).join(), "unknown");
});
test("per-kind confirmation: degraded needs 5 frames", () => {
  const cf = { jammer: 3, unknown: 6, degraded: 5, clear: 3 };
  assert.strictEqual(run(new AlertGate({ confirmFrames: cf }), rep("degraded", 4)).length, 0);
  assert.strictEqual(run(new AlertGate({ confirmFrames: cf }), rep("degraded", 5)).length, 1);
});
const APP = { jammer: 1, unknown: 2, degraded: 5, clear: 3 };     // the settings the dashboard really uses
test("APP settings: a jammer sounds on the very FIRST frame it is seen (instant)", () => {
  const s = run(new AlertGate({ confirmFrames: APP }), [null, null, "jammer", "jammer", "jammer"]);
  assert.strictEqual(s.length, 1); assert.strictEqual(s[0][1], 2);          // frame index 2 = first jammer frame
});
test("APP settings: one borderline UNKNOWN frame is silent, two in a row sound", () => {
  assert.strictEqual(run(new AlertGate({ confirmFrames: APP }), [null, null, "unknown", null, null, null]).length, 0);
  const s = run(new AlertGate({ confirmFrames: APP }), [null, null, "unknown", "unknown", "unknown"]);
  assert.strictEqual(s.length, 1); assert.strictEqual(s[0][1], 3);
});
test("APP settings: an instant jammer still sounds only ONCE during 200 frames and again only after the cool-down", () => {
  const g = new AlertGate({ confirmFrames: APP, cooldownMs: 8000 });
  assert.strictEqual(run(g, [...rep("jammer", 200)]).length, 1);
  const g2 = new AlertGate({ confirmFrames: APP, cooldownMs: 8000 });
  const first = run(g2, [...rep("jammer", 3), ...rep(null, 3)], 0);                 // ends at ~1 s
  const soon = run(g2, [...rep("jammer", 3), ...rep(null, 3)], 1000);              // 1 s later: inside the cool-down (then clears)
  const later = run(g2, rep("jammer", 3), 20000);                                  // 20 s later
  assert.deepStrictEqual([first.length, soon.length, later.length], [1, 0, 1]);
});
test("APP settings: UNKNOWN then jammer (a signal changing) gives one jammer alert, not an extra UNKNOWN", () => {
  const s = run(new AlertGate({ confirmFrames: APP }), [null, null, "unknown", "jammer", "jammer", "jammer"]);
  assert.deepStrictEqual(s.map(x => x[0]), ["jammer"]);
});
function fake(rate, silent, total) { return { sampleRate: rate, getChannelData: () => { const a = new Float32Array(total); a.fill(0.5, silent); return a; } }; }
test("leadingSilence: skips silence at the start of a sound (keeps 2 ms of lead-in)", () => {
  const v = leadingSilence(fake(44100, 2340, 20000));
  assert.ok(Math.abs(v - (2340 - 88) / 44100) < 1e-9, "got " + v);
});
test("leadingSilence: 0 when the sound starts at once, 0 when nothing is audible in the first 250 ms", () => {
  assert.strictEqual(leadingSilence(fake(44100, 0, 20000)), 0);
  assert.strictEqual(leadingSilence(fake(44100, 15000, 20000)), 0);
});
test("every pattern used by the app exists", () => {
  ["jammer", "unknown", "degraded", "test", "on", "blip"].forEach(k => assert.ok(PATTERNS[k] && PATTERNS[k].length, k));
});

console.log("\n" + passed + " passed, " + failed + " failed");
process.exit(failed ? 1 : 0);
