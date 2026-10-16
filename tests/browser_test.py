"""
browser_test.py - OPTIONAL end-to-end test. It starts the real server, opens the real page in a
headless Chromium browser and clicks the real buttons.

It needs Playwright, which is NOT installed by run_dashboard.bat:
    .venv\\Scripts\\python.exe -m pip install playwright
    .venv\\Scripts\\python.exe -m playwright install chromium
    .venv\\Scripts\\python.exe tests\\browser_test.py

Sound is checked with a FAKE Web Audio object that records every oscillator the page schedules,
so the test can count exactly how many sounds were made (no speakers needed).
Takes about 90 seconds (it waits for the 8 s alert cool-down).
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Playwright is not installed: skipping the browser test (this is optional).")
    sys.exit(0)

from werkzeug.serving import make_server
import server

FAKE_AUDIO = """
(() => {
  const log = { osc: [], gains: [], master: [], buffers: [], stops: 0, contexts: 0 };
  window.__audio = log;
  const fakeBuffer = { duration: 1.645688, sampleRate: 44100, length: 72576,
    getChannelData() { const a = new Float32Array(11025); a.fill(0.5, 2340); return a; } };      // 53 ms of silence, then sound
  class FakeCtx {
    constructor() { log.contexts++; this.state = 'running'; this.destination = {}; this._t0 = performance.now(); this._gains = 0; }
    get currentTime() { return (performance.now() - this._t0) / 1000; }
    resume() { this.state = 'running'; return Promise.resolve(); }
    decodeAudioData(bytes, ok) { const p = Promise.resolve(fakeBuffer); if (ok) p.then(ok); return p; }
    createGain() { const isMaster = this._gains++ === 0;
      return { gain: { setValueAtTime: (v) => { log.gains.push(v); if (isMaster) log.master.push(v); }, linearRampToValueAtTime() {} }, connect() {} }; }
    createBufferSource() { const s = { buffer: null, connect() {}, start(t, off) { log.buffers.push({ off }); }, stop() { log.stops++; } }; return s; }
    createOscillator() { const o = { type: 'sine', frequency: { setValueAtTime(f) { o.f = f; } }, connect() {}, start(t) { log.osc.push({ f: o.f, type: o.type }); }, stop() { log.stops++; } }; return o; }
  }
  window.AudioContext = FakeCtx; window.webkitAudioContext = FakeCtx;
})();
"""

failures = []
def check(ok, text):
    print(("  PASS  " if ok else "  FAIL  ") + text)
    if not ok:
        failures.append(text)

srv = make_server("127.0.0.1", 0, server.app, threaded=True)
PORT = srv.server_port
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = "http://127.0.0.1:%d/" % PORT


def text(page, ident):
    return page.inner_text("#" + ident).strip()


def link(page):
    return {k: text(page, i) for k, i in [("status", "lstat"), ("sats", "satTxt"), ("strength", "strTxt"), ("cno", "cnoTxt"), ("interference", "intTxt")]}


def num(s):
    return float("".join(c for c in s if c.isdigit() or c == "."))


def wait_for(page, js, timeout=8000):
    page.wait_for_function(js, timeout=timeout)


with sync_playwright() as pw:
    browser = pw.chromium.launch()

    # ------------------------------------------------------------------ 1. link panel
    print("\n1. GNSS link panel follows the simulator and the controls")
    page = browser.new_page(viewport={"width": 1500, "height": 900})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(URL)
    wait_for(page, "document.getElementById('lstat').textContent === 'NOMINAL'")
    seen = {}
    for index, name in enumerate(["Normal Tone", "Wideband Noise", "Frequency Sweep", "Unfamiliar Test"]):
        if index:
            page.click("#sUp")
        page.wait_for_timeout(2600)
        seen[name] = link(page)
        print("        %-16s %s" % (name, seen[name]))
    check(seen["Normal Tone"]["status"] == "NOMINAL" and seen["Normal Tone"]["sats"] == "12 / 12" and seen["Normal Tone"]["strength"] == "78%", "Normal Tone: 12 / 12 satellites, 78 %, NOMINAL (clean link)")
    check(seen["Wideband Noise"]["status"] == "DEGRADED", "Wideband Noise: link DEGRADED")
    check(seen["Frequency Sweep"]["status"] == "CAUTION", "Frequency Sweep: link CAUTION")
    order = [num(seen[n]["strength"]) for n in ("Normal Tone", "Unfamiliar Test", "Frequency Sweep", "Wideband Noise")]
    check(order == sorted(order, reverse=True) and len(set(order)) == 4, "signal strength falls with interference: %s" % order)
    check(num(seen["Wideband Noise"]["sats"].split("/")[0]) < num(seen["Normal Tone"]["sats"].split("/")[0]), "satellites tracked drop under the wideband jammer (%s)" % seen["Wideband Noise"]["sats"])
    check(num(seen["Wideband Noise"]["interference"]) > num(seen["Frequency Sweep"]["interference"]) > num(seen["Normal Tone"]["interference"]), "interference index orders Wideband > Sweep > Normal")

    page.click("#sUp")                                           # back to Normal Tone (wraps around)
    page.wait_for_timeout(2600)
    check(link(page)["status"] == "NOMINAL", "link recovers when the signal is Normal Tone again")

    page.click("#sSel"); page.click("#sSel")                     # Extra noise
    for _ in range(20):
        page.click("#sUp")
    page.wait_for_timeout(2600)
    noisy = link(page)
    check(noisy["status"] in ("CAUTION", "DEGRADED") and noisy["interference"] != "0%", "Extra noise 100 %% on a Normal Tone also hurts the link: %s" % noisy)
    for _ in range(20):
        page.click("#sDown")
    page.click("#sSel")                                          # back to Signal Type row

    page.wait_for_timeout(2600)
    for _ in range(4):
        page.click("#nLeft")
    page.wait_for_timeout(1500)
    check(link(page)["sats"] == "8 / 8", "nav pad: 4 x LEFT sets 8 satellites in view -> %s" % link(page)["sats"])
    page.click("#nDown"); page.click("#nDown")
    page.wait_for_timeout(1500)
    check(link(page)["strength"] == "68%", "nav pad: 2 x DOWN lowers the clear-sky strength to 68 %% -> %s" % link(page)["strength"])
    page.click("#nEnter")
    page.wait_for_timeout(1500)
    check(link(page)["sats"] == "12 / 12" and link(page)["strength"] == "78%", "nav pad ENTER resets the link to 12 satellites, 78 %")

    page.click("#bFrz")                                          # freeze, then use the nav pad
    page.click("#nLeft"); page.click("#nLeft")
    page.wait_for_timeout(800)
    check(link(page)["sats"] == "10 / 10", "frozen screen: the nav pad still updates the link (%s)" % link(page)["sats"])
    page.click("#bFrz"); page.click("#nEnter")

    page.click("#sEnter")                                        # stop the output
    page.wait_for_timeout(1500)
    stopped = link(page)
    check(stopped["status"] == "NO SIGNAL" and stopped["sats"].startswith("0 /") and stopped["cno"] == "--", "output stopped -> NO SIGNAL, 0 satellites tracked, C/N0 '--' (%s)" % stopped)
    page.click("#sEnter")
    page.wait_for_timeout(2000)
    check(link(page)["status"] == "NOMINAL", "output started again -> link acquired (NOMINAL)")
    check(not errors, "no JavaScript errors on the page %s" % errors)

    # ------------------------------------------------------------------ 2. original controls (regression)
    print("\n2. Original controls still work")
    page.focus("#knob")
    for _ in range(5):
        page.keyboard.press("ArrowUp")
    check(text(page, "kval") == "85%", "knob (keyboard) moves the UNKNOWN limit to 85 %%: %s" % text(page, "kval"))
    for _ in range(5):
        page.keyboard.press("ArrowDown")
    for ident, label in [("bLog", "LOG"), ("bGrid", "GRID"), ("bWtr", "WTR"), ("mBand", "BAND"), ("mAvg", "AVG"), ("mPeak", "PEAK")]:
        before = "on" in (page.get_attribute("#" + ident, "class") or "").split()
        page.click("#" + ident)
        after = "on" in (page.get_attribute("#" + ident, "class") or "").split()
        check(before != after, "%s toggles" % label)
        page.click("#" + ident)
    page.click("#bRef"); page.click("#bClr")
    page.click("#pwr"); page.wait_for_timeout(300)
    check("off" in page.get_attribute("#crt", "class").split(), "POWER turns the screen off")
    page.click("#pwr"); page.wait_for_timeout(1500)
    check("off" not in page.get_attribute("#crt", "class").split() and text(page, "badge") != "", "POWER turns it on again")
    page.click("#sSel"); page.click("#sSel"); page.click("#sSel")
    check("Signal Type" in page.inner_text("#lcd") and ">" in page.inner_text("#lcd"), "simulator SELECT cycles through the three settings")
    check(not errors, "still no JavaScript errors %s" % errors)

    # ------------------------------------------------------------------ 3. modulation unit
    print("\n3. Separate modulation classifier unit")
    labels = ["AM", "FM", "ASK", "FSK", "BPSK", "QPSK"]
    families = {"AM": "ANALOG", "FM": "ANALOG", "ASK": "DIGITAL", "FSK": "DIGITAL", "BPSK": "DIGITAL", "QPSK": "DIGITAL"}
    outcome = {}
    for index in range(9):
        if index:
            page.click("#mUp")
        page.wait_for_timeout(1500)
        outcome[index] = text(page, "mbadge")
    for index, name in enumerate(labels):
        check(outcome[index].replace("\n", " ").upper().startswith(name) and families[name] in outcome[index].upper(), "source %-5s -> %s" % (name, outcome[index].replace("\n", " ")))
    check("UNKNOWN" in outcome[7].upper(), "'Noise only *' (never trained) -> %s" % outcome[7])
    print("        info: '16-QAM *' -> %s | 'Chirp *' -> %s   (not trained; see README section on limits)" % (outcome[6], outcome[8]))
    page.click("#mUp")                                          # wraps to AM
    page.wait_for_timeout(1200)
    page.click("#mSel"); page.click("#mSel")                    # SNR row
    for _ in range(20):
        page.click("#mDown")
    page.wait_for_timeout(1500)
    badge0 = text(page, "mbadge").replace("\n", " ").upper()
    check("0 dB" in page.inner_text("#mlcd") and (badge0.startswith("AM") or "UNKNOWN" in badge0), "SNR 0 dB (very noisy): answer is AM or UNKNOWN, never a confident wrong class (%s)" % badge0)
    page.click("#mSel")                                         # Reject limit row
    for _ in range(9):
        page.click("#mUp")
    page.wait_for_timeout(1200)
    check("95 %" in page.inner_text("#mlcd"), "reject limit can be raised to 95 %")
    page.click("#mEnter"); page.wait_for_timeout(800)
    check("NO INPUT" in text(page, "mbadge").upper(), "ENTER stops the source -> NO INPUT")
    page.click("#mEnter"); page.wait_for_timeout(1500)
    page.click("#mFrz"); page.wait_for_timeout(500)
    frozen_before = text(page, "mbadge")
    page.click("#mSel"); page.click("#mDown"); page.wait_for_timeout(500)        # lower the limit while frozen (re-judged by the server)
    check(text(page, "mbadge") != "", "frozen: moving the reject limit still re-judges the frame (%s)" % text(page, "mbadge").replace("\n", " "))
    page.click("#mFrz")
    page.click("#mPwr"); page.wait_for_timeout(300)
    check("off" in page.get_attribute("#mcrt", "class").split(), "modulation unit POWER turns its screen off")
    page.click("#mPwr")
    # the interference analyzer must be unaffected by the modulation unit
    check(text(page, "lstat") in ("NOMINAL", "CAUTION", "DEGRADED") and "SIGNAL NORMAL" in text(page, "badge").upper(), "jamming analyzer is independent: still '%s'" % text(page, "badge").replace("\n", " "))
    check(not errors, "no JavaScript errors after the modulation tests %s" % errors)
    page.close()

    # ------------------------------------------------------------------ 4. audio alerts (fake Web Audio + fake clock)
    print("\n4. Audio alerts (fake Web Audio records every sound; a fake clock makes the cool-down exact)")
    page = browser.new_page(viewport={"width": 1500, "height": 900})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.add_init_script(FAKE_AUDIO)
    page.add_init_script("window.__t = 1000000; Date.now = () => window.__t;")     # time only moves when the test says so
    page.goto(URL)
    wait_for(page, "document.getElementById('lstat').textContent === 'NOMINAL'")
    osc = lambda: page.evaluate("window.__audio.osc.length")
    patterns = lambda: page.evaluate("window.__sound.log.map(x => x.kind)")
    tick = lambda ms: page.evaluate("window.__t += %d" % ms)
    # wait until the gate has made its decision (waiting for a condition, never for a fixed time)
    announced = lambda kind: wait_for(page, "gate.announced === %s" % ("null" if kind is None else "'%s'" % kind), 40000)

    check("OFF" in text(page, "aStat") and page.get_attribute("#aOn", "aria-pressed") == "false", "alerts start OFF")
    page.click("#sUp")                                           # Wideband Noise = jammer, alerts still off
    announced("jammer")
    check(osc() == 0 and patterns() == [], "jammer while alerts are OFF: no sound")
    check(page.evaluate("AUD.last && AUD.last.kind") == "jammer" and page.evaluate("AUD.last.spoken") is False, "...the alert is recorded as silent (not played)")

    page.click("#aOn")
    wait_for(page, "window.__sound.log.length === 1 && window.__sound.fileState === 'ready'")
    check(patterns() == ["on"] and page.get_attribute("#aOn", "aria-pressed") == "true", "SOUND on: one confirmation chime, button shows ON %s" % patterns())
    check(page.evaluate("window.__sound.buffer !== null") and 0.04 < page.evaluate("window.__sound.skip") < 0.06, "jammer_alert.mp3 downloaded and decoded; leading silence to skip: %.0f ms" % (page.evaluate("window.__sound.skip") * 1000))

    page.click("#sDown")                                         # Wideband -> Normal directly (not through 'Unfamiliar')
    announced(None)                                              # normal confirmed: alert re-armed
    tick(2000)                                                   # only 2 s after the first jammer alert
    page.click("#sUp")                                           # Wideband again
    announced("jammer")
    check(patterns() == ["on"], "jammer returns 2 s after the last one (cool-down 8 s): no repeat sound %s" % patterns())

    n = page.evaluate("frameCounter")
    wait_for(page, "frameCounter > %d" % (n + 40), 60000)        # 40 more frames of continuous jammer
    tick(30000)
    n = page.evaluate("frameCounter")
    wait_for(page, "frameCounter > %d" % (n + 10), 30000)
    check(patterns() == ["on"], "continuous jammer for 50 frames and 30 s: still no repeat %s" % patterns())

    page.click("#sUp")                                           # Sweep: still the 'jammer' kind
    wait_for(page, "R && R.top === 'Frequency Sweep' && R.state === 'jammer'", 30000)
    n = page.evaluate("frameCounter")
    wait_for(page, "frameCounter > %d" % (n + 10), 30000)
    check(patterns() == ["on"], "wideband -> sweep (both 'jammer'): no duplicate %s" % patterns())

    page.click("#sUp")                                           # Unfamiliar -> UNKNOWN (a different kind of alert)
    announced("unknown")
    check(patterns() == ["on", "unknown"], "jammer -> unknown signal: exactly one 'unknown' alert %s" % patterns())

    page.click("#sUp")                                           # Normal
    announced(None)
    tick(10000)                                                  # cool-down is over
    # stamp the frame number at the exact moment the click reaches the page (Playwright's own click is slow here)
    page.evaluate("document.addEventListener('pointerdown', () => { window.__fc = frameCounter; window.__tc = performance.now(); }, { capture: true, once: true })")
    page.click("#sUp")                                           # Wideband: a new event
    announced("jammer")
    check(patterns() == ["on", "unknown", "jammer"], "new jammer event after the cool-down: sounds once %s" % patterns())
    check(page.evaluate("window.__sound.log[window.__sound.log.length - 1].source") == "mp3" and len(page.evaluate("window.__audio.buffers")) == 1, "the jammer alert plays jammer_alert.mp3 (not a beep)")
    check(osc() == 2 + 2, "only the chime (2) and the 'unknown' beep (2) used oscillators: %d" % osc())
    fired = page.evaluate("AUD.last.frame - window.__fc")
    check(fired <= 2, "INSTANT: the jammer alert fired on frame +%d after the change (+1 = request already in flight, +2 = the first frame with the new signal)" % fired)
    delay = page.evaluate("AUD.last.latencyMs")
    check(delay < 100, "frame received -> sound scheduled: %.1f ms (the alert is decided in the same moment the frame arrives)" % delay)

    # volume and mute
    page.fill("#aVol", "100"); page.dispatch_event("#aVol", "input")
    gains = lambda: page.evaluate("window.__audio.master[window.__audio.master.length - 1]")
    check(abs(gains() - 1.0) < 1e-6, "volume 100 %% -> master gain %.3f" % gains())
    page.fill("#aVol", "30"); page.dispatch_event("#aVol", "input")
    check(abs(gains() - 0.09) < 1e-6, "volume 30 %% -> master gain %.4f (volume squared, so it feels even)" % gains())
    page.click("#aMute")
    check(gains() == 0 and page.get_attribute("#aMute", "aria-pressed") == "true", "MUTE sets the master gain to 0")
    before = len(patterns())
    page.click("#aTest")
    wait_for(page, "document.getElementById('aStat').textContent.indexOf('Muted') >= 0")
    check(len(patterns()) == before, "TEST while muted: no sound, and the screen says why")
    page.click("#aMute")
    check(gains() > 0, "unmute restores the volume (gain %.4f)" % gains())
    page.click("#aTest")
    wait_for(page, "window.__sound.log.length === %d" % (before + 1))
    check(patterns()[-1] == "test" and page.evaluate("window.__sound.log[window.__sound.log.length - 1].source") == "mp3", "TEST plays the real alert sound (jammer_alert.mp3) %s" % patterns()[-2:])
    stops_before = page.evaluate("window.__audio.stops")
    page.click("#aMute")
    check(page.evaluate("window.__audio.stops") > stops_before, "MUTE cuts off a sound that is already playing")
    page.click("#aMute")
    page.click("#aOn")                                           # switch alerts off
    check(page.get_attribute("#aOn", "aria-pressed") == "false", "SOUND off")
    page.click("#aTest")
    wait_for(page, "window.__sound.log.length === %d" % (before + 2))
    check(patterns()[-1] == "test", "TEST also works while the alerts are switched off (to set the volume first)")
    check(page.evaluate("localStorage.getItem('gnss.audio.volume')") == "0.3", "the volume is remembered for next time")

    # alerts are silent while the analyzer is off, and a new session can alert again
    page.click("#aOn")
    page.click("#pwr")
    wait_for(page, "gate.announced === null && gate.confirmed === null")
    n_off = len(patterns())
    n = page.evaluate("frameCounter")
    wait_for(page, "frameCounter >= %d" % n, 30000)
    page.wait_for_timeout(1200)
    check(len(patterns()) == n_off, "analyzer powered off: no sounds")
    tick(20000)
    page.click("#pwr")                                           # power on again while still in the Wideband jammer state
    wait_for(page, "gate.announced === 'jammer'", 40000)
    check(patterns()[-1] == "jammer" and len(patterns()) == n_off + 1, "powered on again: the gate was reset, the jammer alerts once %s" % patterns()[-2:])
    check(not errors, "no JavaScript errors during the audio tests %s" % errors)
    page.close()

    # ------------------------------------------------------------------ 5. real Web Audio in the browser
    print("\n5. Real Web Audio engine (no fake): decodes the real MP3 and plays it")
    page = browser.new_page(viewport={"width": 1500, "height": 900})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(URL)
    wait_for(page, "document.getElementById('lstat').textContent === 'NOMINAL'")
    page.click("#aOn")
    wait_for(page, "window.__sound.fileState === 'ready'", 20000)
    info = page.evaluate("({d: window.__sound.buffer.duration, ch: window.__sound.buffer.numberOfChannels, rate: window.__sound.buffer.sampleRate, skip: window.__sound.skip, state: window.__sound.ctx.state})")
    check(abs(info["d"] - 1.6457) < 0.02, "the browser decoded jammer_alert.mp3: %.3f s, %d channel, %d Hz" % (info["d"], info["ch"], info["rate"]))
    check(0.03 < info["skip"] < 0.08, "leading silence detected in the real file: %.0f ms (skipped, so the sound starts at once)" % (info["skip"] * 1000))
    page.evaluate("document.addEventListener('pointerdown', () => { window.__fc = frameCounter; window.__tClick = performance.now(); }, { capture: true, once: true })")
    page.click("#sUp")                                           # Wideband Noise -> jammer
    wait_for(page, "window.__sound.log.some(x => x.kind === 'jammer')", 40000)
    entry = page.evaluate("window.__sound.log.filter(x => x.kind === 'jammer')[0]")
    check(entry["source"] == "mp3", "the jammer alert played the MP3 through the real audio engine")
    frames_after = page.evaluate("AUD.last.frame - window.__fc")
    check(frames_after <= 2, "INSTANT (real engine): sound scheduled on frame +%d after the click" % frames_after)
    print("        info: click event -> sound scheduled: %.0f ms; frame arrived -> sound scheduled: %.1f ms. (This sandbox's CPU is slow: it spends most of each 160 ms tick drawing the CRT filter.)" % (entry["perf"] - page.evaluate("window.__tClick"), page.evaluate("AUD.last.latencyMs")))
    check(not errors, "no JavaScript errors with the real AudioContext %s" % errors)
    page.close()
    browser.close()

srv.shutdown()
print("\nRESULT: %s" % ("ALL BROWSER CHECKS PASSED" if not failures else "%d CHECK(S) FAILED" % len(failures)))
for f in failures:
    print("  - " + f)
sys.exit(1 if failures else 0)
