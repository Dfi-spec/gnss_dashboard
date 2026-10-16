/*
 * alerts.js - optional audio alerts for the GNSS dashboard.
 *
 * Two parts:
 *   AlertGate   decides WHEN an alert may sound (pure logic, no audio, tested with Node).
 *   AlertSound  plays the sound with the browser's Web Audio API.
 *               - JAMMER alert (and the TEST button): static/audio/jammer_alert.mp3 (decoded in advance, so it
 *                 starts immediately; any silence at the very start of the file is skipped automatically).
 *               - UNKNOWN / LINK DEGRADED alerts and the small confirmation chime: short synthesized beeps.
 *               - If the MP3 is missing or cannot be decoded, the jammer alert falls back to a synthesized siren.
 *
 * Duplicate-alert prevention (all in AlertGate):
 *   1. Confirm      a state must be seen for a few frames in a row before it counts. The app feeds the gate with
 *                   the single-frame decision (not the smoothed one), so confirmation is fast: jammer 1 frame,
 *                   unknown 2, degraded 5.
 *   2. One per event an alert sounds once when the state is entered, not on every frame while it lasts.
 *   3. Re-arm       the alert can only sound again after the state has cleared (back to normal).
 *   4. Cooldown     the same kind of alert cannot sound again within `cooldownMs`, even after it cleared.
 *   5. Priority     jammer > unknown > degraded. A lower-priority alert is silent while a higher one is
 *                   active, and for `holdMs` after it (the link status lags behind a jammer).
 *   6. Queue        two sounds never overlap: a new one waits until the previous one has finished
 *                   (at most 2 waiting, the rest are dropped).
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.GnssAlerts = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var PRIORITY = { jammer: 3, unknown: 2, degraded: 1 };

  function AlertGate(options) {
    options = options || {};
    // confirmFrames: one number for every kind, or {jammer: 1, unknown: 2, degraded: 5, clear: 3}
    var cf = options.confirmFrames === undefined ? 3 : options.confirmFrames;
    this.confirmFrames = typeof cf === "number" ? { jammer: cf, unknown: cf, degraded: cf, clear: cf } : cf;
    this.cooldownMs = options.cooldownMs === undefined ? 8000 : options.cooldownMs;
    this.holdMs = options.holdMs === undefined ? 4000 : options.holdMs;
    this.reset();
  }

  AlertGate.prototype.reset = function () {
    this.candidate = null;       // what the last frames looked like (null = nothing to alert about)
    this.candidateCount = 0;
    this.confirmed = null;       // the state that has been confirmed
    this.announced = null;       // the alert kind already sounded for the current event
    this.lastFired = {};         // kind -> time it last sounded
    this.lastKind = null;        // most recent kind that sounded
    this.lastKindAt = -Infinity;
  };

  /* feed(kind, nowMs): kind is "jammer", "unknown", "degraded" or null.
     Returns the kind that should sound now, or null. Call once per NEW frame. */
  AlertGate.prototype.feed = function (kind, now) {
    kind = kind || null;
    if (kind === this.candidate) this.candidateCount += 1;
    else { this.candidate = kind; this.candidateCount = 1; }
    var need = this.confirmFrames[kind === null ? "clear" : kind] || 3;
    if (this.candidateCount < need) return null;                    // 1. not confirmed yet

    this.confirmed = kind;
    if (kind === null) { this.announced = null; return null; }      // 3. cleared: re-arm
    if (kind === this.announced) return null;                       // 2. already sounded for this event

    // a new confirmed alert state: decide whether it may sound
    var blocked = false;
    var sinceSame = now - (this.lastFired[kind] === undefined ? -Infinity : this.lastFired[kind]);
    if (sinceSame < this.cooldownMs) blocked = true;                // 4. cooldown
    if (this.lastKind && PRIORITY[kind] < PRIORITY[this.lastKind] && now - this.lastKindAt < this.holdMs) blocked = true;  // 5. priority hold
    this.announced = kind;                                          // consumed either way: no replay later
    if (blocked) return null;
    this.lastFired[kind] = now;
    this.lastKind = kind;
    this.lastKindAt = now;
    return kind;
  };

  /* ---------------------------------------------------------------- sound */
  // synthesized patterns. each step: [frequency Hz, seconds, wave type]; a gap is [0, seconds]
  var PATTERNS = {
    jammer:   [[880, 0.16, "square"], [0, 0.06], [660, 0.16, "square"], [0, 0.06], [880, 0.16, "square"], [0, 0.06], [660, 0.16, "square"]],   // only used if the MP3 is unavailable
    unknown:  [[440, 0.20, "triangle"], [0, 0.05], [554, 0.28, "triangle"]],
    degraded: [[330, 0.18, "sine"], [0, 0.08], [330, 0.18, "sine"]],
    test:     [[523, 0.14, "sine"], [0, 0.03], [659, 0.14, "sine"], [0, 0.03], [784, 0.22, "sine"]],                                              // only used if the MP3 is unavailable
    on:       [[660, 0.10, "sine"], [0, 0.02], [880, 0.14, "sine"]],
    blip:     [[740, 0.08, "sine"]]
  };
  var FILE_KINDS = { jammer: true, test: true };      // these play the MP3 when it is ready
  var MAX_GAIN = 0.35;                                 // volume 100 % (for the beeps), kept moderate on purpose
  var FILE_GAIN = 1.6;                                 // the MP3 is a voice recording and quieter than the beeps

  // seconds of silence at the start of a decoded buffer (looks at the first 250 ms only)
  function leadingSilence(buffer) {
    var data = buffer.getChannelData(0), limit = Math.min(data.length, Math.floor(buffer.sampleRate * 0.25));
    for (var i = 0; i < limit; i++) {
      if (Math.abs(data[i]) > 0.01) return Math.max(0, i - Math.floor(buffer.sampleRate * 0.002)) / buffer.sampleRate;
    }
    return 0;
  }

  function AlertSound() {
    this.ctx = null;
    this.master = null;
    this.volume = 0.5;
    this.muted = false;
    this.nextFree = 0;           // audio-clock time when the queue is free again
    this.queued = 0;
    this.active = [];            // sounds that are playing or scheduled (so MUTE can cut them off)
    this.raw = null;             // downloaded MP3 bytes, waiting for the audio context
    this.buffer = null;          // decoded MP3
    this.skip = 0;               // leading silence to skip (seconds)
    this.fileState = "none";     // none | loading | fetched | ready | missing | error
    this.log = [];               // [{kind, source, at, perf}] of every sound really scheduled (used by tests)
  }

  AlertSound.prototype.supported = function () {
    return typeof window !== "undefined" && !!(window.AudioContext || window.webkitAudioContext);
  };

  // download the MP3 now (needs no user click); it is decoded as soon as the audio context exists
  AlertSound.prototype.preload = function (url) {
    var self = this;
    if (typeof fetch === "undefined") { this.fileState = "missing"; return Promise.resolve(); }
    this.fileState = "loading";
    return fetch(url).then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.arrayBuffer(); })
      .then(function (bytes) { self.raw = bytes; self.fileState = "fetched"; return self._decode(); })
      .catch(function () { self.fileState = "missing"; });
  };

  AlertSound.prototype._decode = function () {
    var self = this;
    if (!this.raw || !this.ctx || this.buffer) return Promise.resolve();
    var bytes = this.raw; this.raw = null;
    return new Promise(function (ok, bad) { var p = self.ctx.decodeAudioData(bytes, ok, bad); if (p && p.then) p.then(ok, bad); })
      .then(function (buffer) { self.buffer = buffer; self.skip = leadingSilence(buffer); self.fileState = "ready"; })
      .catch(function () { self.fileState = "error"; });
  };

  AlertSound.prototype.ensure = function () {
    if (this.ctx) return this.ctx;
    var C = window.AudioContext || window.webkitAudioContext;
    if (!C) return null;
    this.ctx = new C();
    this.master = this.ctx.createGain();
    this.master.connect(this.ctx.destination);
    this._applyGain();
    this._decode();
    return this.ctx;
  };

  AlertSound.prototype._applyGain = function () {
    if (!this.master) return;
    var level = this.muted ? 0 : this.volume * this.volume;        // squared: feels linear to the ear
    this.master.gain.setValueAtTime(level, this.ctx.currentTime);
  };

  // call from a click: browsers only allow sound after a user gesture
  AlertSound.prototype.enable = function () {
    var ctx = this.ensure();
    if (!ctx) return Promise.resolve(false);
    var self = this;
    return (ctx.state === "suspended" ? ctx.resume() : Promise.resolve())
      .then(function () { return self._decode(); })
      .then(function () { return true; }, function () { return false; });
  };

  AlertSound.prototype.setVolume = function (v) { this.volume = Math.max(0, Math.min(1, v)); this._applyGain(); };
  AlertSound.prototype.setMuted = function (m) {
    this.muted = !!m; this._applyGain();
    if (this.muted) this.stopAll();                                 // MUTE also cuts off a sound that is playing
  };
  AlertSound.prototype.stopAll = function () {
    this.active.forEach(function (node) { try { node.stop(0); } catch (e) {} });
    this.active = []; this.queued = 0;
    if (this.ctx) this.nextFree = this.ctx.currentTime;
  };

  AlertSound.prototype._reserve = function (now) {                    // 6. queue: never overlap two sounds
    if (this.nextFree > now && this.queued >= 2) return null;
    var start = Math.max(now + 0.005, this.nextFree);                 // nothing is added to the delay when the queue is free
    if (start > now + 0.03) {
      this.queued += 1;
      var self = this;
      setTimeout(function () { self.queued = Math.max(0, self.queued - 1); }, Math.max(0, (start - now) * 1000));
    }
    return start;
  };

  AlertSound.prototype._playFile = function (kind) {
    var ctx = this.ctx, start = this._reserve(ctx.currentTime);
    if (start === null) return false;
    var src = ctx.createBufferSource(), boost = ctx.createGain();
    src.buffer = this.buffer;
    boost.gain.setValueAtTime(FILE_GAIN, start);
    src.connect(boost); boost.connect(this.master);
    src.start(start, this.skip);
    this.active.push(src);
    this.nextFree = start + (this.buffer.duration - this.skip) + 0.15;
    this.log.push({ kind: kind, source: "mp3", at: Date.now(), perf: performance.now() });
    return true;
  };

  /* play(kind): returns true if a sound was scheduled. */
  AlertSound.prototype.play = function (kind) {
    var pattern = PATTERNS[kind];
    if (!pattern || this.muted) return false;
    var ctx = this.ensure();
    if (!ctx) return false;
    if (ctx.state === "suspended") ctx.resume();
    if (FILE_KINDS[kind] && this.buffer) return this._playFile(kind);
    var start = this._reserve(ctx.currentTime);
    if (start === null) return false;
    var t = start;
    for (var i = 0; i < pattern.length; i++) {
      var step = pattern[i], freq = step[0], dur = step[1];
      if (freq > 0) {
        var osc = ctx.createOscillator(), env = ctx.createGain();
        osc.type = step[2] || "sine";
        osc.frequency.setValueAtTime(freq, t);
        env.gain.setValueAtTime(0.0001, t);
        env.gain.linearRampToValueAtTime(MAX_GAIN, t + 0.012);        // short fade in/out: no clicks
        env.gain.setValueAtTime(MAX_GAIN, t + Math.max(0.012, dur - 0.03));
        env.gain.linearRampToValueAtTime(0.0001, t + dur);
        osc.connect(env); env.connect(this.master);
        osc.start(t); osc.stop(t + dur + 0.02);
        this.active.push(osc);
      }
      t += dur;
    }
    this.nextFree = t + 0.12;
    this.log.push({ kind: kind, source: "synth", at: Date.now(), perf: performance.now() });
    return true;
  };

  return { AlertGate: AlertGate, AlertSound: AlertSound, PATTERNS: PATTERNS, PRIORITY: PRIORITY, MAX_GAIN: MAX_GAIN, leadingSilence: leadingSilence };
});
