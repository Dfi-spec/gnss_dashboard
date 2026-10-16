"""
selftest.py - checks that the dashboard backend really uses your model.
Run:  run_selftest.bat      (or:  python tests/selftest.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import ml_model
import server

client = server.app.test_client()
failures = []


def check(ok, text):
    print(("  PASS  " if ok else "  FAIL  ") + text)
    if not ok:
        failures.append(text)


print("\n1. Status endpoint")
status = client.get("/api/status").get_json()
check(status["classes"] == list(server.CLF.model.classes_), "classes come from signal_model.joblib: %s" % status["classes"])
check(status["n_estimators"] == 100, "model has %d trees" % status["n_estimators"])
check(status["dataset"]["rows"] == 3000, "signal_dataset.csv has %d rows" % status["dataset"]["rows"])
print("        hold-out accuracy on the csv split: %.1f%%" % (status["dataset"]["holdout_accuracy"] * 100))

print("\n2. /api/step returns exactly what your model says for the same signal")
names = status["signal_types"]
for index, name in enumerate(names):
    reply = client.post("/api/step", json={"type": index, "seed": 42, "extra": 0, "threshold": 0.8, "history": []}).get_json()
    x = np.array([reply["features"][k] for k in ("peak_frequency", "bandwidth", "spectral_flatness", "rms")]).reshape(1, -1)
    direct = server.CLF.model.predict_proba(x)[0]
    same = np.allclose(direct, reply["probabilities"], atol=1e-9)
    check(same, "%-24s -> %-16s (%.1f%%), probabilities identical to model.predict_proba" % (name, reply["prediction"], reply["confidence"] * 100))

print("\n3. /api/classify (features in, classification out)")
sample = client.post("/api/step", json={"type": 1, "seed": 7, "extra": 0, "threshold": 0.8}).get_json()
f = sample["features"]
reply = client.post("/api/classify", json={"features": [f["peak_frequency"], f["bandwidth"], f["spectral_flatness"], f["rms"]], "threshold": 0.8}).get_json()
check(reply["prediction"] == "Wideband Noise", "features of a Wideband Noise signal -> %s" % reply["prediction"])
check(client.post("/api/classify", json={"features": [1, 2]}).status_code == 400, "bad input is rejected with 400")

print("\n4. Behaviour over 60 live frames per signal (unknown limit 80%)")
for index, name in enumerate(names):
    counts = {}
    for seed in range(1, 61):
        r = client.post("/api/step", json={"type": index, "seed": seed, "extra": 0, "threshold": 0.8, "history": []}).get_json()
        counts[r["prediction"]] = counts.get(r["prediction"], 0) + 1
    print("        %-24s %s" % (name, counts))
    if name != "Unfamiliar Test Signal":
        expected = names[index] if name == "Normal Tone" else ("Wideband Noise" if name == "Wideband Noise" else "Frequency Sweep")
        # A rare borderline frame (measured: about 1 in 7,000 sweep frames and 1 in 20,000 wideband frames) has the RIGHT
        # class at 74-77 % confidence, which the 80 % rule turns into UNKNOWN. One such frame is tolerated; a WRONG class never is.
        wrong = {k: v for k, v in counts.items() if k not in (expected, "UNKNOWN")}
        check(not wrong and counts.get(expected, 0) >= 59, "%s is classified as %s (no wrong class; at most 1 borderline UNKNOWN frame tolerated)" % (name, expected))
    else:
        check(counts.get("UNKNOWN", 0) >= 55, "Unfamiliar Test Signal is flagged UNKNOWN in %d of 60 frames" % counts.get("UNKNOWN", 0))

print("\n5. Controls change the result")
quiet = client.post("/api/step", json={"type": 0, "seed": 42, "extra": 0, "threshold": 0.8}).get_json()
noisy = client.post("/api/step", json={"type": 0, "seed": 42, "extra": 100, "threshold": 0.8}).get_json()
check(noisy["features"]["rms"] > quiet["features"]["rms"], "Extra noise 100%% raises RMS (%.3f -> %.3f)" % (quiet["features"]["rms"], noisy["features"]["rms"]))
strict = client.post("/api/step", json={"type": 3, "seed": 42, "threshold": 0.95}).get_json()
loose = client.post("/api/step", json={"type": 3, "seed": 42, "threshold": 0.50}).get_json()
check(strict["prediction"] == "UNKNOWN" and loose["prediction"] != "UNKNOWN", "Unknown limit knob: 95%% -> %s, 50%% -> %s" % (strict["prediction"], loose["prediction"]))
check(client.post("/api/step", json={"type": 9}).status_code == 400, "invalid signal type is rejected")

print("\n6. Simulated GNSS link (new /api/link and the link inside /api/step)")
import link_model
def settled(type_index, extra=0, sats=12, strength=78, frames=25):
    prev, last = None, None
    for _ in range(frames):
        last = client.post("/api/step", json={"type": type_index, "seed": 42, "extra": extra, "threshold": 0.8,
                                              "link": {"sats": sats, "strength": strength, "prev_cno": prev}}).get_json()["link"]
        prev = last["cno_trend"]
    return last
normal, wide, sweep, unfam = settled(0), settled(1), settled(2), settled(3)
check(normal["status"] == "NOMINAL" and normal["sats_tracked"] == 12 and normal["strength"] == 78, "Normal Tone: 12 of 12 satellites, 78 %%, NOMINAL (%s)" % normal["status"])
check(wide["status"] == "DEGRADED" and wide["sats_tracked"] < 12, "Wideband Noise: DEGRADED, %d of 12 satellites tracked" % wide["sats_tracked"])
check(sweep["status"] == "CAUTION", "Frequency Sweep: CAUTION (%d of 12, %d %%)" % (sweep["sats_tracked"], sweep["strength"]))
order = [normal["strength"], unfam["strength"], sweep["strength"], wide["strength"]]
check(order == sorted(order, reverse=True) and len(set(order)) == 4, "signal strength falls as interference grows: %s" % order)
noisy = settled(0, extra=100)
check(noisy["interference"] > 0.5 and noisy["strength"] < normal["strength"], "Extra noise 100 %% on a Normal Tone lowers the link (strength %d %%)" % noisy["strength"])
base = settled(0, sats=8, strength=60)
check(base["sats_in_view"] == 8 and base["strength"] == 60, "nav-pad baseline (8 satellites, 60 %%) is honoured: %d / %d, %d %%" % (base["sats_tracked"], base["sats_in_view"], base["strength"]))
stopped = client.post("/api/link", json={"running": False, "interference": 0, "link": {"sats": 12, "strength": 78}}).get_json()
check(stopped["status"] == "NO SIGNAL" and stopped["sats_tracked"] == 0 and stopped["cno_dbhz"] is None, "stopped output -> NO SIGNAL, 0 satellites tracked")
first = client.post("/api/step", json={"type": 1, "seed": 42, "link": {"sats": 12, "strength": 78, "prev_cno": 42.0}}).get_json()["link"]
check(first["cno_trend"] > wide["cno_trend"] + 3, "link changes gradually, not in one jump (C/N0 trend %.1f after one frame, settles near %.1f)" % (first["cno_trend"], wide["cno_trend"]))
check(link_model.status_for(5, 80) == "CAUTION" and link_model.status_for(4, 80) == "DEGRADED" and link_model.status_for(12, 24) == "DEGRADED" and link_model.status_for(12, 49) == "CAUTION",
      "status thresholds are the same as the original app.py (<=4 / <25 DEGRADED, <=7 / <50 CAUTION)")
check(client.post("/api/link", json={"interference": "abc"}).status_code == 400, "bad /api/link input is rejected with 400")
check(client.post("/api/step", json={"type": 0, "seed": 1, "link": {"sats": 999, "strength": -5}}).get_json()["link"]["sats_in_view"] == 20, "out-of-range nav-pad values are clamped (999 satellites -> 20)")

print("\n7. Modulation classifier (separate model: modulation_model.joblib)")
import modulation_engine as mod
mstatus = client.get("/api/mod/status")
check(mstatus.status_code == 200, "/api/mod/status answers")
mstatus = mstatus.get_json()
check(sorted(mstatus["classes"]) == sorted(mod.MOD_CLASSES) and server.MOD.model.n_estimators == 300, "classes: %s (%d trees)" % (mstatus["classes"], server.MOD.model.n_estimators))
check(mstatus["classes"] != status["classes"], "it is a different model from the interference classifier")
print("        hold-out accuracy on modulation_dataset.csv (synthetic, SNR 0-30 dB): %.1f%%" % (mstatus["holdout_accuracy"] * 100))
X, y, Xt, yt, snr_t = mod.load_dataset_csv()
check(len(X) == 3600 and len(Xt) == 1200, "dataset file has %d training and %d test rows" % (len(X), len(Xt)))
check(float(np.mean(server.MOD.model.predict(Xt) == yt)) >= 0.97, "re-measured accuracy on the held-out rows: %.1f%% (needs >= 97%%)" % (np.mean(server.MOD.model.predict(Xt) == yt) * 100))
for index, name in enumerate(mod.MOD_CLASSES):
    reply = client.post("/api/mod/step", json={"type": index, "seed": 42, "snr": 20, "threshold": 0.7}).get_json()
    direct = server.MOD.model.predict_proba(np.array(reply["feature_vector"]).reshape(1, -1))[0]
    check(reply["prediction"] == name and reply["family"] == mod.FAMILY[name] and np.allclose(direct, reply["probabilities"], atol=1e-9),
          "%-5s -> %-5s (%s, %.1f%%), probabilities identical to model.predict_proba" % (name, reply["prediction"], reply["family"], reply["confidence"] * 100))
print("        60 live frames per type, SNR 20 dB, 20 different variations:")
for index, name in enumerate(mod.MOD_CLASSES):
    counts = {}
    for seed in range(1, 61):
        r = client.post("/api/mod/step", json={"type": index, "seed": (seed % 20) + 1, "snr": 20, "threshold": 0.7, "history": []}).get_json()
        counts[r["prediction"]] = counts.get(r["prediction"], 0) + 1
    print("        %-5s %s" % (name, counts))
    check(counts.get(name, 0) >= 54, "%s is recognised in %d of 60 frames (needs >= 54, i.e. 90%%)" % (name, counts.get(name, 0)))
noise_only = [client.post("/api/mod/step", json={"type": 7, "seed": s, "snr": 20, "threshold": 0.7}).get_json()["prediction"] for s in range(1, 41)]
check(noise_only.count("UNKNOWN") >= 36, "'Noise only' (never trained) is UNKNOWN in %d of 40 frames (needs >= 36)" % noise_only.count("UNKNOWN"))
qam = [client.post("/api/mod/step", json={"type": 6, "seed": s, "snr": 20, "threshold": 0.7}).get_json()["prediction"] for s in range(1, 41)]
chirp = [client.post("/api/mod/step", json={"type": 8, "seed": s, "snr": 20, "threshold": 0.7}).get_json()["prediction"] for s in range(1, 41)]
print("        info only (limits of the UNKNOWN rule): 16-QAM -> UNKNOWN in %d of 40 frames, otherwise %s; Chirp -> UNKNOWN in %d of 40, otherwise %s" % (
    qam.count("UNKNOWN"), sorted(set(qam) - {"UNKNOWN"}), chirp.count("UNKNOWN"), sorted(set(chirp) - {"UNKNOWN"})))
low = [client.post("/api/mod/step", json={"type": 4, "seed": s, "snr": 0, "threshold": 0.7}).get_json()["prediction"] for s in range(1, 41)]
check(sum(1 for p in low if p not in ("BPSK", "UNKNOWN")) <= 4, "BPSK at 0 dB: %d of 40 frames are a confident wrong class (needs <= 4)" % sum(1 for p in low if p not in ("BPSK", "UNKNOWN")))
strict = client.post("/api/mod/step", json={"type": 0, "seed": 42, "snr": 5, "threshold": 0.95}).get_json()
check(client.post("/api/mod/classify", json={"features": strict["feature_vector"], "threshold": 0.5}).get_json()["prediction"] == "AM", "/api/mod/classify re-judges a frame from its features")
check(client.post("/api/mod/classify", json={"features": [1, 2]}).status_code == 400, "bad /api/mod/classify input is rejected with 400")
check(client.post("/api/mod/step", json={"type": 99}).status_code == 400, "invalid modulation type is rejected")
check(client.post("/api/step", json={"type": 0, "seed": 42}).get_json()["prediction"] == "Normal Tone", "the interference classifier is unchanged and still answers")

print("\n8. Audio alert logic (static/alerts.js, needs Node.js)")
import shutil, subprocess
node = shutil.which("node")
if node:
    run = subprocess.run([node, str(Path(__file__).resolve().parent / "test_alert_gate.js")], capture_output=True, text=True)
    lines = [l for l in run.stdout.splitlines() if l.strip()]
    print("\n".join("        " + l.strip() for l in lines[-3:]))
    check(run.returncode == 0, "duplicate-alert prevention tests (tests/test_alert_gate.js)")
else:
    print("        SKIPPED: Node.js is not installed. The alert logic is also covered by tests/browser_test.py (optional).")

print("\n9. Instant alert decision and the jammer alert sound file")
for index, name in enumerate(names):
    r = client.post("/api/step", json={"type": index, "seed": 42, "extra": 0, "threshold": 0.8, "history": []}).get_json()
    check("instant" in r and r["instant"]["status"] == r["status"], "%s: single-frame decision (instant) equals the screen decision when there is no history (%s)" % (name, r["instant"]["status"]))
# a signal that just changed: the screen is still averaging old frames, the instant decision already sees the new signal
old_frames = [client.post("/api/step", json={"type": 0, "seed": s, "extra": 0, "threshold": 0.8, "history": []}).get_json()["frame_probabilities"] for s in (1, 2, 3)]
switched = client.post("/api/step", json={"type": 1, "seed": 5, "extra": 0, "threshold": 0.8, "history": old_frames}).get_json()
check(switched["instant"]["status"] == "jammer" and switched["status"] == "unknown", "first frame after Normal -> Wideband: instant = jammer, smoothed screen = %s (this is the delay the alert no longer waits for)" % switched["status"])
mp3 = client.get("/static/audio/jammer_alert.mp3")
check(mp3.status_code == 200 and mp3.mimetype == "audio/mpeg" and len(mp3.data) > 10000, "static/audio/jammer_alert.mp3 is served (%d bytes, %s)" % (len(mp3.data), mp3.mimetype))

print("\n10. Your interference-classifier files are untouched (byte-identical to the originals)")
import hashlib
ORIGINAL_SHA256 = {'signal_model.joblib': '2b88c6707597b867c8b4dcd6b3937219539e48af28ec9340844bb68a59b47cc4', 'signal_dataset.csv': '670819dc6f2384e3d9144564d7d0a7d27d5d5d12c26341e29ad89b69cd7246c3', 'ml_model.py': '7715f3720fdbe796874d0315e739c0e8f062d42a7c9d95be2ae4408a0533dd3d', 'signal_engine.py': 'bcab896a9224547a01f80d2674c0b2c06d11a028d67a8b623a4e9e7419437402', 'app.py': '7f5a307c865406a43249ccc867c18727a7e2181a44e637248aad3dc47b0a77a9', 'signal_simulator.py': '67478c4fd13513db502cca63f82680282db3cf4788ae9fd48944ae5bd9bf3d2c'}
for filename, expected in ORIGINAL_SHA256.items():
    actual = hashlib.sha256((Path(__file__).resolve().parents[1] / filename).read_bytes()).hexdigest()
    check(actual == expected, "%-22s unchanged" % filename)

print("\nAll data above is synthetic. This does not validate real-world GNSS jamming detection.")
print("\nRESULT: %s" % ("ALL CHECKS PASSED" if not failures else "%d CHECK(S) FAILED" % len(failures)))
sys.exit(1 if failures else 0)
