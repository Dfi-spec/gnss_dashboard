
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="GNSS Signal Monitor",
    page_icon="📡",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "signal_model.joblib"

SAMPLE_RATE = 1000
DURATION = 1.0
SAMPLES = int(SAMPLE_RATE * DURATION)
t = np.arange(SAMPLES) / SAMPLE_RATE


# ==========================================
# LOAD SAVED MODEL
# ==========================================

@st.cache_resource
def load_saved_model():
    saved = joblib.load(MODEL_PATH)

    # Supports the latest model bundle and the older
    # model-only file format.
    if isinstance(saved, dict) and "model" in saved:
        model = saved["model"]
        threshold = saved.get("unknown_threshold", 0.70)
    else:
        model = saved
        threshold = 0.70

    return model, threshold


# ==========================================
# FEATURE EXTRACTION
# Must match ml_model.py
# ==========================================

def extract_features(signal):
    spectrum = np.abs(rfft(signal))
    frequencies = rfftfreq(
        len(signal),
        d=1 / SAMPLE_RATE
    )

    peak_frequency = frequencies[np.argmax(spectrum)]

    if np.max(spectrum) > 0:
        active = frequencies[
            spectrum >= 0.1 * np.max(spectrum)
        ]
        bandwidth = (
            active[-1] - active[0]
            if len(active) > 0 else 0.0
        )
    else:
        bandwidth = 0.0

    power = spectrum**2 + 1e-12
    spectral_flatness = (
        np.exp(np.mean(np.log(power)))
        / np.mean(power)
    )

    rms = np.sqrt(np.mean(signal**2))

    return [
        float(peak_frequency),
        float(bandwidth),
        float(spectral_flatness),
        float(rms)
    ]


# ==========================================
# SIGNAL GENERATOR
# ==========================================

def generate_signal(signal_type, seed):
    rng = np.random.default_rng(seed)

    if signal_type == "Normal Tone":
        frequency = rng.uniform(30, 120)
        amplitude = rng.uniform(0.7, 1.2)
        noise = rng.uniform(0.01, 0.10)

        signal = (
            amplitude * np.sin(2 * np.pi * frequency * t)
            + rng.normal(0, noise, SAMPLES)
        )

    elif signal_type == "Wideband Noise":
        signal = rng.normal(
            0, rng.uniform(0.3, 0.8), SAMPLES
        )

    elif signal_type == "Frequency Sweep":
        start = rng.uniform(10, 50)
        end = rng.uniform(100, 200)
        rate = (end - start) / DURATION

        phase = 2 * np.pi * (
            start * t + 0.5 * rate * t**2
        )

        signal = (
            np.sin(phase)
            + rng.normal(0, 0.05, SAMPLES)
        )

    elif signal_type == "Unfamiliar Test Signal":
        signal = (
            0.7 * np.sin(2 * np.pi * 65 * t)
            + 0.4 * np.sin(2 * np.pi * 175 * t)
            + rng.normal(0, 0.25, SAMPLES)
        )

    else:
        raise ValueError("Unsupported signal type")

    return signal


# ==========================================
# DASHBOARD HEADER
# ==========================================

st.title("📡 GNSS Signal Classification Monitor")
st.caption(
    "Machine-learning demonstration | "
    "Synthetic signal data only — not connected to a live GNSS receiver"
)

st.warning(
    "DEMO MODE: All signals and satellite indicators on this "
    "dashboard are simulated. This is not a validated real-world "
    "GNSS jamming detector."
)


# ==========================================
# LOAD MODEL SAFELY
# ==========================================

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Make sure signal_model.joblib "
        "is in the same Downloads folder as app.py."
    )
    st.stop()

try:
    model, threshold = load_saved_model()
except Exception as error:
    st.error(f"Could not load the saved model: {error}")
    st.info(
        "Run py -3.13 ml_model.py to regenerate the latest "
        "model file, then refresh this dashboard."
    )
    st.stop()


# ==========================================
# SIDEBAR CONTROLS
# ==========================================

st.sidebar.header("Simulation Controls")

signal_type = st.sidebar.selectbox(
    "Choose a test signal",
    [
        "Normal Tone",
        "Wideband Noise",
        "Frequency Sweep",
        "Unfamiliar Test Signal"
    ]
)

seed = st.sidebar.slider(
    "Signal variation",
    min_value=1,
    max_value=100,
    value=42
)

satellites = st.sidebar.slider(
    "Simulated satellites visible",
    min_value=0,
    max_value=20,
    value=12
)

signal_strength = st.sidebar.slider(
    "Simulated signal strength",
    min_value=0,
    max_value=100,
    value=78,
    help="Illustrative indicator only; not a measured RF value."
)

st.sidebar.caption(
    "Changing these controls changes the simulation, "
    "not real satellite signals."
)


# ==========================================
# GENERATE AND CLASSIFY SIGNAL
# ==========================================

signal = generate_signal(signal_type, seed)
features = extract_features(signal)

probabilities = model.predict_proba(
    np.array(features).reshape(1, -1)
)[0]

best_index = int(np.argmax(probabilities))
confidence = float(probabilities[best_index])

if confidence < threshold:
    prediction = "UNKNOWN"
else:
    prediction = str(model.classes_[best_index])


# ==========================================
# SIMULATED STATUS
# ==========================================

if satellites <= 4 or signal_strength < 25:
    gnss_status = "DEGRADED"
    status_color = "🔴"
elif satellites <= 7 or signal_strength < 50:
    gnss_status = "CAUTION"
    status_color = "🟡"
else:
    gnss_status = "NOMINAL"
    status_color = "🟢"


# ==========================================
# TOP METRICS
# ==========================================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Model prediction", prediction)
col2.metric("Highest class probability", f"{confidence:.1%}")
col3.metric("Simulated satellites", f"{satellites} / 20")
col4.metric("Simulated signal strength", f"{signal_strength}%")

st.subheader(f"{status_color} Simulated GNSS status: {gnss_status}")

if prediction == "UNKNOWN":
    st.warning(
        "The highest class probability is below the rejection "
        "threshold. The model returned UNKNOWN."
    )
else:
    st.info(
        f"Best matching trained class: {prediction}. "
        "This is a synthetic classification, not proof of "
        "jamming or interference."
    )

st.progress(satellites / 20)
st.caption("Satellite indicator is simulated and does not come from GPS hardware.")


# ==========================================
# SIGNAL VISUALIZATIONS
# ==========================================

st.subheader("Signal Analysis")

left, right = st.columns(2)

with left:
    st.markdown("**Time-domain waveform**")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, signal, linewidth=0.8)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.markdown("**Frequency spectrum**")

    spectrum = np.abs(rfft(signal))
    frequencies = rfftfreq(
        len(signal),
        d=1 / SAMPLE_RATE
    )

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(frequencies, spectrum, linewidth=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# EXTRACTED FEATURES
# ==========================================

st.subheader("Extracted Signal Features")

feature_col1, feature_col2, feature_col3, feature_col4 = (
    st.columns(4)
)

feature_col1.metric("Peak frequency", f"{features[0]:.1f} Hz")
feature_col2.metric("Bandwidth", f"{features[1]:.1f} Hz")
feature_col3.metric("Spectral flatness", f"{features[2]:.4f}")
feature_col4.metric("RMS amplitude", f"{features[3]:.4f}")


# ==========================================
# MODEL CLASS PROBABILITIES
# ==========================================

st.subheader("Model Class Probabilities")

for class_name, probability in zip(
    model.classes_,
    probabilities
):
    st.write(f"**{class_name}** — {probability:.1%}")
    st.progress(float(probability))

st.caption(
    f"UNKNOWN is returned when the highest class probability "
    f"is below {threshold:.0%}. Model probabilities can be "
    "overconfident for unfamiliar signals."
)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Project prototype | Python + Streamlit + Random Forest | "
    "Synthetic data only | No live RF capture or hardware validation"
)
