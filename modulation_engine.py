"""
modulation_engine.py - a SEPARATE classifier for AM / FM and digital modulation.

It does not touch the interference classifier (ml_model.py / signal_model.joblib).
It has its own simulated signals, its own features, its own Random Forest
(modulation_model.joblib) and its own dataset (modulation_dataset.csv).

Classes the model is trained on
  analog  : AM, FM
  digital : ASK, FSK, BPSK, QPSK

Signals are SYNTHETIC (generated here, sampled at 8000 samples/s, 2048 samples each,
a carrier between 1500 and 2500 Hz plus Gaussian noise). Nothing here reads a real radio.
A result is a statement about these simulated signals, not about real-world radio.

Features (all computed from the one frame, no knowledge of the true type):
  envelope   : env_std, env_kurt, env_skew, env_line   (amplitude behaviour)
  frequency  : if_std, if_kurt, if_line, if_diff_kurt  (instantaneous-frequency behaviour)
  statistics : c20, c40, c42                           (normalised cumulants of the baseband)
  phase      : p1, p2, p4                              (phase coherence at 1x, 2x, 4x the phase)
  spectrum   : bw95, flat                              (occupied bandwidth, flatness)
"""
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.fft import rfft, rfftfreq, fft
from scipy.signal import hilbert, butter, sosfiltfilt
from scipy.stats import kurtosis, skew

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modulation_model.joblib"
DATASET_PATH = BASE_DIR / "modulation_dataset.csv"
METRICS_PATH = BASE_DIR / "modulation_metrics.json"

FS = 8000
N = 2048
T = np.arange(N) / FS
MOD_CLASSES = ["AM", "FM", "ASK", "FSK", "BPSK", "QPSK"]           # what the model is trained on
UNSEEN_TYPES = ["16-QAM (unseen)", "Noise only (unseen)", "Chirp (unseen)"]   # NOT in training
SIM_TYPES = MOD_CLASSES + UNSEEN_TYPES
FAMILY = {"AM": "analog", "FM": "analog", "ASK": "digital", "FSK": "digital", "BPSK": "digital", "QPSK": "digital"}
FEATURE_NAMES = ["env_std", "env_kurt", "env_skew", "env_line", "if_std", "if_kurt", "if_line",
                 "if_diff_kurt", "c20", "c40", "c42", "p1", "p2", "p4", "bw95", "flat"]
DEFAULT_THRESHOLD = 0.70
TRAIN_SNR_RANGE = (0.0, 30.0)     # dB, what the dataset covers
EDGE = 120                        # samples dropped at both ends (filter edge effects)


# --------------------------------------------------------------------------------------
# Signal generator
# --------------------------------------------------------------------------------------
def _smooth(x, width=6):
    kernel = np.ones(width) / width
    return np.convolve(x, kernel, mode="same")


def _symbol_stream(rng, rate, levels):
    """Random symbols held (rectangular pulses) for 1/rate seconds; both ends of the alphabet appear."""
    n_sym = int(np.ceil(N * rate / FS)) + 3
    symbols = rng.integers(0, levels, n_sym)
    symbols[0], symbols[1] = 0, levels - 1               # make sure the alphabet is really used
    offset = rng.uniform(0, 1)
    index = np.floor(T * rate + offset).astype(int)
    return symbols[index], symbols


def generate_clean(name, seed):
    """One noise-free frame of the named signal. `seed` fixes its parameters (carrier, depth, rate...)."""
    rng = np.random.default_rng(int(seed))
    fc = rng.uniform(1500, 2500)
    theta = rng.uniform(0, 2 * np.pi)

    if name == "AM":
        depth = rng.uniform(0.3, 0.9)
        f1, f2 = rng.uniform(30, 100), rng.uniform(100, 180)
        message = 0.75 * np.cos(2 * np.pi * f1 * T + rng.uniform(0, 6.28)) + 0.25 * np.cos(2 * np.pi * f2 * T + rng.uniform(0, 6.28))
        return (1 + depth * message) * np.cos(2 * np.pi * fc * T + theta)

    if name == "FM":
        fm = rng.uniform(40, 120)
        deviation = rng.uniform(300, 900)
        phase = (deviation / fm) * np.sin(2 * np.pi * fm * T + rng.uniform(0, 6.28))
        return np.cos(2 * np.pi * fc * T + phase + theta)

    if name == "ASK":
        rate = rng.uniform(100, 300)
        bits, _ = _symbol_stream(rng, rate, 2)
        low = rng.uniform(0.0, 0.3)
        amplitude = _smooth(np.where(bits == 1, 1.0, low))
        return amplitude * np.cos(2 * np.pi * fc * T + theta)

    if name == "FSK":
        rate = rng.uniform(100, 300)
        deviation = rng.uniform(300, 700)
        bits, _ = _symbol_stream(rng, rate, 2)
        freq = _smooth(np.where(bits == 1, 1.0, -1.0)) * deviation
        phase = 2 * np.pi * np.cumsum(freq) / FS
        return np.cos(2 * np.pi * fc * T + phase + theta)

    if name in ("BPSK", "QPSK", "16-QAM (unseen)"):
        rate = rng.uniform(100, 300)
        if name == "BPSK":
            ids, _ = _symbol_stream(rng, rate, 2)
            baseband = np.exp(1j * np.pi * ids)
        elif name == "QPSK":
            ids, _ = _symbol_stream(rng, rate, 4)
            baseband = np.exp(1j * (np.pi / 4 + np.pi / 2 * ids))
        else:  # 16-QAM: not in the training set
            ids_i, _ = _symbol_stream(rng, rate, 4)
            ids_q, _ = _symbol_stream(rng, rate, 4)
            level = np.array([-3.0, -1.0, 1.0, 3.0])
            baseband = (level[ids_i] + 1j * level[ids_q]) / np.sqrt(10) * 1.2
        baseband = _smooth(baseband.real) + 1j * _smooth(baseband.imag)
        return np.real(baseband * np.exp(1j * (2 * np.pi * fc * T + theta)))

    if name == "Noise only (unseen)":
        noise = np.random.default_rng(int(seed)).normal(0, 1, N)
        band = butter(4, [max(fc - 600, 50) / (FS / 2), min(fc + 600, 3900) / (FS / 2)], btype="band", output="sos")
        return sosfiltfilt(band, noise)

    if name == "Chirp (unseen)":
        span = rng.uniform(600, 1400)
        start = fc - span / 2
        rate = span / (N / FS)
        return np.cos(2 * np.pi * (start * T + 0.5 * rate * T ** 2) + theta)

    raise ValueError("Unknown modulation type: %s" % name)


def add_noise(clean, snr_db, rng):
    power = float(np.mean(clean ** 2))
    sigma = np.sqrt(power / (10 ** (snr_db / 10.0)))
    return clean + rng.normal(0.0, sigma, clean.shape)


def generate_signal(name, seed, snr_db, rng=None):
    """Noisy frame. The noise is fresh on every call (like a live input)."""
    rng = rng if rng is not None else np.random.default_rng()
    return add_noise(generate_clean(name, seed), float(snr_db), rng)


# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------
_SOS = butter(6, 1500 / (FS / 2), output="sos")


def _estimate_carrier(z):
    power = np.abs(np.fft.fft(z)) ** 2
    freqs = np.fft.fftfreq(N, 1 / FS)
    keep = (freqs > 0)
    f, p = freqs[keep], power[keep]
    fc = np.sum(f * p) / np.sum(p)
    for half_width in (1500.0, 1200.0):                       # re-centre on the signal, ignore far noise
        window = np.abs(f - fc) <= half_width
        fc = np.sum(f[window] * p[window]) / np.sum(p[window])
    return fc


def _line_ratio(values):
    """Strength of the strongest spectral line of `values` compared with the typical level (log10)."""
    spectrum = np.abs(rfft(values - np.mean(values)))
    lo, hi = 2, int(1000 * N / FS)
    band = spectrum[lo:hi]
    return float(np.log10(np.max(band) / (np.median(band) + 1e-12) + 1.0))


def analyze(x):
    """Returns (features, traces). traces = envelope and instantaneous frequency for the display."""
    x = np.asarray(x, dtype=float)
    z = hilbert(x - np.mean(x))
    fc = _estimate_carrier(z)
    base = z * np.exp(-2j * np.pi * fc * T)
    base = sosfiltfilt(_SOS, base.real) + 1j * sosfiltfilt(_SOS, base.imag)
    base = base[EDGE:-EDGE]
    n = len(base)

    amp = np.abs(base)
    amp_n = amp / (np.mean(amp) + 1e-12)
    env_std = float(np.std(amp_n))
    env_kurt = float(kurtosis(amp_n, fisher=False))
    env_skew = float(skew(amp_n))
    env_line = _line_ratio(amp_n)

    phase = np.unwrap(np.angle(base))
    inst = np.diff(phase) * FS / (2 * np.pi)
    inst = _smooth(inst, 9)[5:-5]
    if_std = float(np.std(inst) / 1000.0)
    if_kurt = float(kurtosis(inst, fisher=False))
    if_line = _line_ratio(inst)
    d_inst = np.diff(inst)
    if_diff_kurt = float(kurtosis(d_inst, fisher=False))

    unit = base / (np.sqrt(np.mean(amp ** 2)) + 1e-12)
    c21 = np.mean(np.abs(unit) ** 2)
    c20 = np.mean(unit ** 2)
    c40 = np.mean(unit ** 4) - 3 * c20 ** 2
    c42 = np.mean(np.abs(unit) ** 4) - np.abs(c20) ** 2 - 2 * c21 ** 2

    angle = np.angle(base)
    weight = np.sum(amp_n) + 1e-12
    p1, p2, p4 = (float(np.max(np.abs(fft(amp_n * np.exp(1j * k * angle)))) / weight) for k in (1, 2, 4))

    spectrum = np.abs(rfft(x * np.hanning(N))) ** 2
    cumulative = np.cumsum(spectrum) / np.sum(spectrum)
    freqs = rfftfreq(N, 1 / FS)
    bw95 = float((freqs[np.searchsorted(cumulative, 0.975)] - freqs[np.searchsorted(cumulative, 0.025)]) / 1000.0)
    flat = float(np.exp(np.mean(np.log(spectrum + 1e-12))) / np.mean(spectrum))

    features = [env_std, env_kurt, env_skew, env_line, if_std, if_kurt, if_line, if_diff_kurt,
                float(np.abs(c20) / (c21 + 1e-12)), float(np.abs(c40) / (c21 ** 2 + 1e-12)), float(np.abs(c42) / (c21 ** 2 + 1e-12)),
                p1, p2, p4, bw95, flat]
    traces = {"envelope": amp_n, "inst_freq_khz": inst / 1000.0, "carrier_hz": float(fc)}
    return features, traces


def extract_features(x):
    return analyze(x)[0]


# --------------------------------------------------------------------------------------
# Dataset file, training, saving and loading
# --------------------------------------------------------------------------------------
def load_dataset_csv(path=DATASET_PATH):
    """Rows of modulation_dataset.csv -> (X_train, y_train, X_test, y_test, snr_test)."""
    parts = {"train": ([], []), "test": ([], [], [])}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != ["split", "snr_db", "label"] + FEATURE_NAMES:
            raise ValueError("modulation_dataset.csv does not have the expected columns")
        for row in reader:
            split, snr, label = row[0], float(row[1]), row[2]
            values = [float(v) for v in row[3:]]
            parts[split][0].append(values)
            parts[split][1].append(label)
            if split == "test":
                parts["test"][2].append(snr)
    return (np.array(parts["train"][0]), np.array(parts["train"][1]),
            np.array(parts["test"][0]), np.array(parts["test"][1]), np.array(parts["test"][2]))


def fit_model(X, y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
    model.fit(X, y)
    return model


def save_bundle(model, metrics=None, path=MODEL_PATH):
    import sklearn
    joblib.dump({
        "model": model,
        "feature_names": FEATURE_NAMES,
        "classes": [str(c) for c in model.classes_],
        "unknown_threshold": DEFAULT_THRESHOLD,
        "sample_rate": FS,
        "samples": N,
        "sklearn_version": sklearn.__version__,
        "metrics": metrics,
    }, path)


# --------------------------------------------------------------------------------------
# The classifier used by the server
# --------------------------------------------------------------------------------------
class ModClassifier:
    """Loads modulation_model.joblib. If it was saved by a different scikit-learn version (or cannot be
    read) the model is rebuilt from modulation_dataset.csv in a few seconds, so the dashboard still starts."""

    def __init__(self, default_threshold=None):
        import sklearn
        self.rebuilt = False
        self.note = None
        bundle = None
        try:
            bundle = joblib.load(MODEL_PATH)
            if bundle.get("sklearn_version") != sklearn.__version__:
                self.note = "modulation_model.joblib was saved with scikit-learn %s, installed is %s: rebuilt from modulation_dataset.csv" % (
                    bundle.get("sklearn_version"), sklearn.__version__)
                bundle = None
        except Exception as error:
            self.note = "could not read modulation_model.joblib (%s): rebuilt from modulation_dataset.csv" % error
        if bundle is None:
            X, y, _, _, _ = load_dataset_csv()
            model = fit_model(X, y)
            metrics = None
            if METRICS_PATH.exists():
                metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            try:
                save_bundle(model, metrics)
            except OSError:
                pass
            bundle = {"model": model, "unknown_threshold": DEFAULT_THRESHOLD, "metrics": metrics}
            self.rebuilt = True
        self.model = bundle["model"]
        self.model.n_jobs = 1
        self.classes = [str(c) for c in self.model.classes_]
        self.saved_threshold = float(bundle.get("unknown_threshold", DEFAULT_THRESHOLD))
        self.default_threshold = self.saved_threshold if default_threshold is None else float(default_threshold)
        self.metrics = bundle.get("metrics")

    def classify_features(self, features, threshold, history=None):
        x = np.asarray(features, dtype=float).reshape(1, -1)
        frame = self.model.predict_proba(x)[0]
        stack = [np.asarray(h, dtype=float) for h in (history or [])[-3:] if len(h) == len(self.classes)]
        stack.append(frame)
        averaged = np.mean(stack, axis=0)
        best = int(np.argmax(averaged))
        confidence = float(averaged[best])
        top = self.classes[best]
        prediction = "UNKNOWN" if confidence < threshold else top
        family = "unknown" if prediction == "UNKNOWN" else FAMILY[top]
        return {
            "classes": self.classes,
            "frame_probabilities": [float(p) for p in frame],
            "probabilities": [float(p) for p in averaged],
            "top_class": top,
            "confidence": confidence,
            "prediction": prediction,
            "family": family,
            "threshold": threshold,
        }
