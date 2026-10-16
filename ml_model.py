
import csv
from pathlib import Path

import joblib
import numpy as np
from scipy.fft import rfft, rfftfreq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. SETTINGS
# ==========================================

OUTPUT_DIR = Path(__file__).resolve().parent

SAMPLE_RATE = 1000
DURATION = 1.0
SAMPLES = int(SAMPLE_RATE * DURATION)
UNKNOWN_THRESHOLD = 0.70

t = np.arange(SAMPLES) / SAMPLE_RATE
rng = np.random.default_rng(42)

SIGNAL_TYPES = [
    "Normal Tone",
    "Wideband Noise",
    "Frequency Sweep"
]

FEATURE_NAMES = [
    "peak_frequency",
    "bandwidth",
    "spectral_flatness",
    "rms"
]

DATASET_PATH = OUTPUT_DIR / "signal_dataset.csv"
MODEL_PATH = OUTPUT_DIR / "signal_model.joblib"


# ==========================================
# 2. GENERATE SYNTHETIC SIGNALS
# ==========================================

def generate_signal(signal_type):
    if signal_type == "Normal Tone":
        frequency = rng.uniform(30, 120)
        amplitude = rng.uniform(0.7, 1.2)
        noise_level = rng.uniform(0.01, 0.10)

        signal = (
            amplitude * np.sin(2 * np.pi * frequency * t)
            + rng.normal(0, noise_level, SAMPLES)
        )

    elif signal_type == "Wideband Noise":
        noise_level = rng.uniform(0.3, 0.8)
        signal = rng.normal(0, noise_level, SAMPLES)

    elif signal_type == "Frequency Sweep":
        start_frequency = rng.uniform(10, 50)
        end_frequency = rng.uniform(100, 200)
        noise_level = rng.uniform(0.01, 0.08)

        sweep_rate = (end_frequency - start_frequency) / DURATION
        phase = 2 * np.pi * (
            start_frequency * t
            + 0.5 * sweep_rate * t**2
        )

        signal = (
            np.sin(phase)
            + rng.normal(0, noise_level, SAMPLES)
        )

    else:
        raise ValueError(f"Unknown signal type: {signal_type}")

    return signal


# ==========================================
# 3. EXTRACT FEATURES
# ==========================================

def extract_features(signal):
    spectrum = np.abs(rfft(signal))
    frequencies = rfftfreq(
        len(signal),
        d=1 / SAMPLE_RATE
    )

    peak_frequency = frequencies[np.argmax(spectrum)]

    if np.max(spectrum) > 0:
        active_frequencies = frequencies[
            spectrum >= 0.1 * np.max(spectrum)
        ]

        if len(active_frequencies) > 0:
            bandwidth = (
                active_frequencies[-1]
                - active_frequencies[0]
            )
        else:
            bandwidth = 0.0
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
# 4. PREDICTION WITH UNKNOWN REJECTION
# ==========================================

def classify_signal(model, signal, threshold=UNKNOWN_THRESHOLD):
    """
    Returns UNKNOWN if the largest model probability
    is below the configured threshold.

    Note: confidence is not a guarantee that a signal
    belongs to a known class.
    """
    extracted = np.array(
        extract_features(signal)
    ).reshape(1, -1)

    probabilities = model.predict_proba(extracted)[0]
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])

    if confidence < threshold:
        predicted_class = "UNKNOWN"
    else:
        predicted_class = str(model.classes_[best_index])

    return predicted_class, confidence, probabilities


# ==========================================
# 5. MAIN PROGRAM
# ==========================================

def main():
    print("=" * 55)
    print(" SIGNAL CLASSIFICATION WITH UNKNOWN REJECTION")
    print("=" * 55)

    features = []
    labels = []
    examples_per_class = 1000

    print("\n[1/7] Generating synthetic dataset...")

    for signal_type in SIGNAL_TYPES:
        print("Generating:", signal_type)

        for _ in range(examples_per_class):
            signal = generate_signal(signal_type)
            features.append(extract_features(signal))
            labels.append(signal_type)

    X = np.array(features, dtype=float)
    y = np.array(labels)

    print("\nTotal signals:", len(X))
    print("Features per signal:", X.shape[1])
    print("Known classes:", SIGNAL_TYPES)


    # ======================================
    # 6. SAVE DATASET
    # ======================================

    print("\n[2/7] Saving dataset...")

    with open(
        DATASET_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(FEATURE_NAMES + ["label"])

        for row, label in zip(X, y):
            writer.writerow(list(row) + [label])

    print("Saved:", DATASET_PATH)


    # ======================================
    # 7. SPLIT DATA
    # ======================================

    print("\n[3/7] Splitting data...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training examples:", len(X_train))
    print("Testing examples:", len(X_test))


    # ======================================
    # 8. TRAIN MODEL
    # ======================================

    print("\n[4/7] Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Training complete.")


    # ======================================
    # 9. SAVE MODEL AND SETTINGS
    # ======================================

    print("\n[5/7] Saving model...")

    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "sample_rate": SAMPLE_RATE,
            "duration": DURATION,
            "unknown_threshold": UNKNOWN_THRESHOLD,
            "signal_types": SIGNAL_TYPES
        },
        MODEL_PATH
    )

    print("Saved trained model:", MODEL_PATH)
    print(f"UNKNOWN threshold: {UNKNOWN_THRESHOLD:.0%}")


    # ======================================
    # 10. EVALUATE KNOWN CLASSES
    # ======================================

    print("\n[6/7] Evaluating known signal classes...")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nKnown-class test accuracy: {accuracy * 100:.2f}%")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=SIGNAL_TYPES,
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print("Class order:", list(model.classes_))
    print(confusion_matrix(y_test, predictions))


    # ======================================
    # 11. TEST UNKNOWN REJECTION
    # ======================================

    print("\n[7/7] Testing UNKNOWN rejection...")

    # Test a synthetic signal that was not one of
    # the three signal types used for training.
    unfamiliar_signal = (
        0.7 * np.sin(2 * np.pi * 65 * t)
        + 0.4 * np.sin(2 * np.pi * 175 * t)
        + rng.normal(0, 0.25, SAMPLES)
    )

    predicted, confidence, probabilities = classify_signal(
        model,
        unfamiliar_signal
    )

    print("\nUnfamiliar synthetic signal test:")
    print("Prediction:", predicted)
    print(f"Highest class probability: {confidence:.2%}")
    print("Class probabilities:")

    for class_name, probability in zip(
        model.classes_,
        probabilities
    ):
        print(f"  {class_name}: {probability:.2%}")

    print("\n" + "=" * 55)
    print(" COMPLETED SUCCESSFULLY")
    print("=" * 55)

    print("\nCreated files:")
    print("1. signal_dataset.csv")
    print("2. signal_model.joblib")

    print("\nImportant:")
    print("- UNKNOWN is a confidence-based rejection rule.")
    print("- Unfamiliar signals may still receive high confidence.")
    print("- All signals in this experiment are synthetic.")
    print("- This is not yet a validated GNSS jamming detector.")


if __name__ == "__main__":
    main()
