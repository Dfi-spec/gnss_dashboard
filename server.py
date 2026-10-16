"""
server.py - local Flask backend for the GNSS Jamming Detection Analyzer dashboard.

Run:   python server.py        (or double-click run_dashboard.bat)
Open:  http://127.0.0.1:5000

The dashboard page (static/index.html) only draws and sends control settings.
Signals are generated here, features are measured here, and the classification
comes from your trained Random Forest (signal_model.joblib) through ml_model.py.

All signals are SYNTHETIC. Nothing here is a validated real-world GNSS
jamming detector.
"""
import csv
import os
import sys
from collections import Counter

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

import signal_engine as eng
import ml_model
import link_model
import modulation_engine as mod

HOST = os.environ.get("GNSS_HOST", "127.0.0.1")      # local machine only
PORT = int(os.environ.get("GNSS_PORT", "5000"))
# The dashboard knob starts here. ml_model.py saves 0.70; 0.80 separates the
# "Unfamiliar Test Signal" from your known classes more reliably. Change it with
#   set GNSS_UNKNOWN_THRESHOLD=0.70      (before running) to match ml_model.py.
DEFAULT_THRESHOLD = float(os.environ.get("GNSS_UNKNOWN_THRESHOLD", "0.80"))

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


def load_classifier():
    try:
        return eng.Classifier(default_threshold=DEFAULT_THRESHOLD)
    except Exception as error:   # most often: scikit-learn version differs from the one that saved the model
        print("\nCould not load signal_model.joblib:", error)
        print("Fix: run retrain_model.bat (rebuilds the dataset and model with the installed scikit-learn).\n")
        sys.exit(1)


CLF = load_classifier()


def load_modulation_classifier():
    """The modulation classifier is separate. If it cannot load, the rest of the dashboard still works."""
    try:
        classifier = mod.ModClassifier()
        if classifier.note:
            print("\nNote:", classifier.note, "\n")
        return classifier
    except Exception as error:
        print("\nThe modulation classifier could not be loaded:", error)
        print("The interference analyzer still works. Run retrain_modulation.bat to rebuild the modulation model.\n")
        return None


MOD = load_modulation_classifier()


def dataset_info():
    """Facts from signal_dataset.csv, and accuracy on the same 20 % hold-out split ml_model.py used."""
    if not eng.DATASET_PATH.exists():
        return None
    rows, labels = [], []
    with open(eng.DATASET_PATH, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            rows.append([float(v) for v in row[:4]])
            labels.append(row[4])
    info = {"rows": len(rows), "class_counts": dict(Counter(labels)), "holdout_accuracy": None}
    try:
        from sklearn.model_selection import train_test_split
        X = np.array(rows)
        y = np.array(labels)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        info["holdout_accuracy"] = float(np.mean(CLF.model.predict(X_test) == y_test))
        info["holdout_rows"] = int(len(y_test))
    except Exception as error:
        info["note"] = "hold-out accuracy not computed: " + str(error)
    return info


DATASET = dataset_info()


def clean_threshold(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = CLF.default_threshold
    return min(0.95, max(0.50, value))


def clean_history(value):
    if not isinstance(value, list):
        return []
    result = []
    for item in value[-3:]:
        if isinstance(item, list) and len(item) == len(CLF.classes):
            try:
                result.append([float(v) for v in item])
            except (TypeError, ValueError):
                pass
    return result


def link_inputs(body):
    """Nav-pad baseline and previous C/N0 sent by the page (validated and clamped)."""
    link = body.get("link") if isinstance(body.get("link"), dict) else {}
    def number(key, default):
        try:
            return float(link.get(key, default))
        except (TypeError, ValueError):
            return default
    prev = link.get("prev_cno")
    try:
        prev = None if prev is None else float(prev)
    except (TypeError, ValueError):
        prev = None
    return {
        "sats_in_view": int(link_model.clamp(number("sats", 12), 0, link_model.MAX_SATS)),
        "strength_percent": int(link_model.clamp(number("strength", 78), 0, 100)),
        "prev_cno": prev,
    }


# ------------------------------------------------------------------ pages
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ------------------------------------------------------------------ API
@app.get("/api/status")
def status():
    import sklearn
    return jsonify({
        "backend": "online",
        "model_file": eng.MODEL_PATH.name,
        "model_type": type(CLF.model).__name__,
        "n_estimators": int(CLF.model.n_estimators),
        "classes": CLF.classes,
        "feature_names": list(ml_model.FEATURE_NAMES),
        "saved_unknown_threshold": CLF.saved_threshold,
        "default_unknown_threshold": CLF.default_threshold,
        "sample_rate": eng.SAMPLE_RATE,
        "samples": eng.SAMPLES,
        "signal_types": eng.SIGNAL_TYPES,
        "scikit_learn": sklearn.__version__,
        "dataset": DATASET,
        "data_source": "synthetic (simulated) signals only",
    })


@app.post("/api/step")
def step():
    """Generate the selected simulated signal, measure features, classify with the model."""
    body = request.get_json(silent=True) or {}
    try:
        type_index = int(body.get("type", 0))
        seed = int(body.get("seed", 42))
        extra = float(body.get("extra", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "type, seed and extra must be numbers"}), 400
    if not 0 <= type_index < len(eng.SIGNAL_TYPES):
        return jsonify({"error": "type must be 0..%d" % (len(eng.SIGNAL_TYPES) - 1)}), 400
    extra = min(100.0, max(0.0, extra))
    threshold = clean_threshold(body.get("threshold"))
    history = clean_history(body.get("history"))

    signal = eng.generate_signal(type_index, seed, extra)
    features = ml_model.extract_features(signal)          # your feature function
    magnitude, lo, hi = eng.spectrum_and_band(signal)
    result = CLF.classify_signal(signal, threshold, history)   # your model

    # the SCREEN smooths over the last few frames (stable, but it needs about 0.6 s to settle on a new signal).
    # Audio alerts must not wait for that, so they use this single-frame decision (same rule, no history).
    _, _, instant_confidence, instant_prediction, instant_status = CLF._decide(result["frame_probabilities"], [], threshold)
    result["instant"] = {"prediction": instant_prediction, "status": instant_status, "confidence": instant_confidence}

    # simulated GNSS link health: nav-pad baseline + the interference this source causes
    interference = link_model.interference_index(type_index, extra, features[3])
    result["link"] = link_model.compute_link(interference, True, **link_inputs(body))

    result.update({
        "signal_type": eng.SIGNAL_TYPES[type_index],
        "samples": np.round(signal, 4).tolist(),
        "spectrum": np.round(magnitude, 3).tolist(),
        "features": {
            "peak_frequency": features[0],
            "bandwidth": features[1],
            "spectral_flatness": features[2],
            "rms": features[3],
            "band_low_bin": lo,
            "band_high_bin": hi,
        },
        "simulated": True,
    })
    return jsonify(result)


@app.post("/api/link")
def link():
    """Recompute the simulated GNSS link without making a new signal.
    The page uses it when the screen is frozen, the output is stopped, or a nav-pad key is pressed.
    body: {"running": true, "interference": 0..1, "link": {"sats": 12, "strength": 78, "prev_cno": null}}"""
    body = request.get_json(silent=True) or {}
    try:
        interference = float(body.get("interference", 0.0))
    except (TypeError, ValueError):
        return jsonify({"error": "interference must be a number between 0 and 1"}), 400
    running = bool(body.get("running", True))
    return jsonify(link_model.compute_link(interference, running, **link_inputs(body)))


# ------------------------------------------------------------------ modulation classifier (separate)
def mod_unavailable():
    return jsonify({"error": "The modulation classifier is not loaded. Run retrain_modulation.bat and restart."}), 503


@app.get("/api/mod/status")
def mod_status():
    if MOD is None:
        return mod_unavailable()
    metrics = MOD.metrics or {}
    return jsonify({
        "backend": "online",
        "model_file": mod.MODEL_PATH.name,
        "n_estimators": int(MOD.model.n_estimators),
        "classes": MOD.classes,
        "families": {name: mod.FAMILY[name] for name in MOD.classes},
        "sim_types": mod.SIM_TYPES,
        "trained_on": mod.MOD_CLASSES,
        "unseen_types": mod.UNSEEN_TYPES,
        "saved_unknown_threshold": MOD.saved_threshold,
        "default_unknown_threshold": MOD.default_threshold,
        "sample_rate": mod.FS,
        "samples": mod.N,
        "rebuilt_at_start": MOD.rebuilt,
        "holdout_accuracy": metrics.get("holdout_accuracy"),
        "accuracy_by_snr": metrics.get("accuracy_by_snr"),
        "unseen_results": metrics.get("unseen_types"),
        "data_source": "synthetic (simulated) signals only",
    })


@app.post("/api/mod/step")
def mod_step():
    """Generate the selected simulated modulated signal, measure features, classify with the modulation model."""
    if MOD is None:
        return mod_unavailable()
    body = request.get_json(silent=True) or {}
    try:
        type_index = int(body.get("type", 0))
        seed = int(body.get("seed", 42))
        snr = float(body.get("snr", 20))
    except (TypeError, ValueError):
        return jsonify({"error": "type, seed and snr must be numbers"}), 400
    if not 0 <= type_index < len(mod.SIM_TYPES):
        return jsonify({"error": "type must be 0..%d" % (len(mod.SIM_TYPES) - 1)}), 400
    snr = min(40.0, max(0.0, snr))
    threshold = clean_threshold(body.get("threshold", MOD.default_threshold))
    history = []
    if isinstance(body.get("history"), list):
        for item in body["history"][-3:]:
            if isinstance(item, list) and len(item) == len(MOD.classes):
                try:
                    history.append([float(v) for v in item])
                except (TypeError, ValueError):
                    pass

    name = mod.SIM_TYPES[type_index]
    frame = mod.generate_signal(name, seed, snr)
    features, traces = mod.analyze(frame)
    result = MOD.classify_features(features, threshold, history)

    spectrum = np.abs(np.fft.rfft(frame * np.hanning(mod.N)))
    spectrum = spectrum / (np.max(spectrum) + 1e-12)
    result.update({
        "signal_type": name,
        "trained_type": name in mod.MOD_CLASSES,
        "snr_db": snr,
        "samples": np.round(frame[:400], 4).tolist(),                     # first 50 ms
        "spectrum": np.round(spectrum, 4).tolist(),                       # 0 .. 4000 Hz
        "envelope": np.round(traces["envelope"][::4], 3).tolist(),
        "inst_freq_khz": np.round(traces["inst_freq_khz"][::4], 3).tolist(),
        "carrier_hz": round(traces["carrier_hz"], 1),
        "features": dict(zip(mod.FEATURE_NAMES, [float(v) for v in features])),
        "feature_vector": [float(v) for v in features],     # in the model's own order (JSON objects are sorted by Flask)
        "simulated": True,
    })
    return jsonify(result)


@app.post("/api/mod/classify")
def mod_classify():
    """Re-judge a frame whose features were already measured (used when the screen is frozen)."""
    if MOD is None:
        return mod_unavailable()
    body = request.get_json(silent=True) or {}
    features = body.get("features")
    if not (isinstance(features, list) and len(features) == len(mod.FEATURE_NAMES)):
        return jsonify({"error": "features must be a list of %d numbers: %s" % (len(mod.FEATURE_NAMES), ", ".join(mod.FEATURE_NAMES))}), 400
    try:
        features = [float(v) for v in features]
    except (TypeError, ValueError):
        return jsonify({"error": "features must be numbers"}), 400
    return jsonify(MOD.classify_features(features, clean_threshold(body.get("threshold", MOD.default_threshold)), []))


@app.post("/api/classify")
def classify():
    """Classify features that were already measured (peak, bandwidth, flatness, rms)."""
    body = request.get_json(silent=True) or {}
    features = body.get("features")
    if not (isinstance(features, list) and len(features) == 4):
        return jsonify({"error": "features must be a list of 4 numbers: peak_frequency, bandwidth, spectral_flatness, rms"}), 400
    try:
        features = [float(v) for v in features]
    except (TypeError, ValueError):
        return jsonify({"error": "features must be numbers"}), 400
    threshold = clean_threshold(body.get("threshold"))
    history = clean_history(body.get("history"))
    return jsonify(CLF.classify_features(features, threshold, history))


if __name__ == "__main__":
    print("=" * 60)
    print(" GNSS Jamming Detection Analyzer - local backend")
    print("=" * 60)
    print(" Model        :", eng.MODEL_PATH.name, "(%d trees)" % CLF.model.n_estimators)
    print(" Classes      :", ", ".join(CLF.classes))
    print(" Unknown limit: saved in model %.0f%%, dashboard starts at %.0f%%" % (CLF.saved_threshold * 100, CLF.default_threshold * 100))
    if DATASET:
        text = " Dataset      : signal_dataset.csv, %d rows" % DATASET["rows"]
        if DATASET.get("holdout_accuracy") is not None:
            text += ", hold-out accuracy %.1f%%" % (DATASET["holdout_accuracy"] * 100)
        print(text)
    if MOD is not None:
        text = " Modulation   : modulation_model.joblib (%d trees), classes %s" % (MOD.model.n_estimators, ", ".join(MOD.classes))
        if MOD.metrics and MOD.metrics.get("holdout_accuracy") is not None:
            text += ", hold-out accuracy %.1f%%" % (MOD.metrics["holdout_accuracy"] * 100)
        print(text)
    else:
        print(" Modulation   : NOT LOADED (run retrain_modulation.bat). The jamming analyzer still works.")
    print(" Data         : SYNTHETIC signals only (not a real GNSS receiver)")
    print("-" * 60)
    print(" Open  http://%s:%d   (press Ctrl+C to stop)" % (HOST, PORT))
    print("=" * 60)
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
