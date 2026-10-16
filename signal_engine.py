"""
signal_engine.py - signal generation and inference for the GNSS dashboard backend.

Everything that decides a classification lives here and uses YOUR project files:
  * ml_model.extract_features()  -> the 4 features (peak frequency, bandwidth,
                                    spectral flatness, RMS)
  * ml_model.classify_signal()   -> probabilities from signal_model.joblib
  * signal_model.joblib          -> your trained Random Forest (loaded with joblib)

app.py (your Streamlit file) cannot be imported because it draws a page when
imported, so its signal generator is mirrored here with the same parameter ranges:
  Normal Tone, Wideband Noise, Frequency Sweep, Unfamiliar Test Signal.
The signal settings (frequency, amplitude, ...) are fixed by the "Variation" number.
The noise is fresh on every call, so the input behaves like a live signal.

All signals are SYNTHETIC. Nothing here reads a real GNSS receiver.
"""
from pathlib import Path

import joblib
import numpy as np
from scipy.fft import rfft

import ml_model  # your own module (unchanged)

SAMPLE_RATE = ml_model.SAMPLE_RATE
SAMPLES = ml_model.SAMPLES
T = ml_model.t
SIGNAL_TYPES = [
    "Normal Tone",
    "Wideband Noise",
    "Frequency Sweep",
    "Unfamiliar Test Signal",
]
MODEL_PATH = Path(ml_model.MODEL_PATH)
DATASET_PATH = Path(ml_model.DATASET_PATH)


# ----------------------------------------------------------------------------
# Signal generator (mirrors generate_signal() in app.py, plus "extra noise")
# ----------------------------------------------------------------------------
def generate_signal(type_index: int, seed: int, extra_noise_percent: float = 0.0):
    """One second of synthetic signal sampled at 1000 samples per second."""
    params = np.random.default_rng(int(seed))   # fixes frequency, amplitude, ...
    live = np.random.default_rng()              # fresh noise on every call
    name = SIGNAL_TYPES[type_index]

    if name == "Normal Tone":
        frequency = params.uniform(30, 120)
        amplitude = params.uniform(0.7, 1.2)
        noise = params.uniform(0.01, 0.10)
        signal = amplitude * np.sin(2 * np.pi * frequency * T) + live.normal(0, noise, SAMPLES)

    elif name == "Wideband Noise":
        signal = live.normal(0, params.uniform(0.3, 0.8), SAMPLES)

    elif name == "Frequency Sweep":
        start = params.uniform(10, 50)
        end = params.uniform(100, 200)
        rate = (end - start) / ml_model.DURATION
        phase = 2 * np.pi * (start * T + 0.5 * rate * T ** 2)
        signal = np.sin(phase) + live.normal(0, 0.05, SAMPLES)

    else:  # Unfamiliar Test Signal (same recipe as ml_model.py / app.py)
        signal = (
            0.7 * np.sin(2 * np.pi * 65 * T)
            + 0.4 * np.sin(2 * np.pi * 175 * T)
            + live.normal(0, 0.25, SAMPLES)
        )

    if extra_noise_percent > 0:
        # 100 % adds noise with a standard deviation of 0.5
        signal = signal + live.normal(0, 0.5 * extra_noise_percent / 100.0, SAMPLES)
    return signal


def spectrum_and_band(signal):
    """FFT magnitude (same as ml_model) and the first/last bin used for bandwidth."""
    magnitude = np.abs(rfft(signal))
    peak = float(np.max(magnitude))
    if peak > 0:
        active = np.nonzero(magnitude >= 0.1 * peak)[0]
        lo, hi = (int(active[0]), int(active[-1])) if len(active) else (-1, -1)
    else:
        lo = hi = -1
    return magnitude, lo, hi


# ----------------------------------------------------------------------------
# Your trained model
# ----------------------------------------------------------------------------
class Classifier:
    def __init__(self, default_threshold=None):
        saved = joblib.load(MODEL_PATH)
        if isinstance(saved, dict) and "model" in saved:
            self.model = saved["model"]
            self.saved_threshold = float(saved.get("unknown_threshold", ml_model.UNKNOWN_THRESHOLD))
        else:
            self.model = saved
            self.saved_threshold = float(ml_model.UNKNOWN_THRESHOLD)
        self.model.n_jobs = 1   # one small sample per request: threads only add delay
        self.classes = [str(c) for c in self.model.classes_]
        self.default_threshold = self.saved_threshold if default_threshold is None else default_threshold

    # -- decision rule: smoothing over the last frames + the UNKNOWN threshold --
    def _decide(self, frame_probs, history, threshold):
        n = len(self.classes)
        stack = [np.asarray(h, dtype=float) for h in (history or [])[-3:] if len(h) == n]
        stack.append(np.asarray(frame_probs, dtype=float))
        averaged = np.mean(stack, axis=0)
        best = int(np.argmax(averaged))
        confidence = float(averaged[best])
        top = self.classes[best]
        prediction = "UNKNOWN" if confidence < threshold else top   # same rule as ml_model.classify_signal
        if prediction == "UNKNOWN":
            status = "unknown"
        elif top == "Normal Tone":
            status = "normal"
        else:
            status = "jammer"
        return averaged, top, confidence, prediction, status

    def _pack(self, frame_probs, history, threshold):
        averaged, top, confidence, prediction, status = self._decide(frame_probs, history, threshold)
        return {
            "classes": self.classes,
            "frame_probabilities": [float(p) for p in frame_probs],
            "probabilities": [float(p) for p in averaged],
            "top_class": top,
            "confidence": confidence,
            "prediction": prediction,
            "status": status,
            "threshold": threshold,
        }

    # -- from a signal (uses ml_model.classify_signal) --
    def classify_signal(self, signal, threshold, history=None):
        _, _, probabilities = ml_model.classify_signal(self.model, signal, threshold)
        return self._pack(probabilities, history, threshold)

    # -- from features that were already measured --
    def classify_features(self, features, threshold, history=None):
        x = np.asarray(features, dtype=float).reshape(1, -1)
        probabilities = self.model.predict_proba(x)[0]
        return self._pack(probabilities, history, threshold)
