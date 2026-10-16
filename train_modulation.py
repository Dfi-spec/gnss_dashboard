"""
train_modulation.py - builds modulation_dataset.csv, trains modulation_model.joblib and writes
modulation_metrics.json for the SEPARATE AM/FM + digital modulation classifier.

Run:  python train_modulation.py            (or double-click retrain_modulation.bat)
It does not change signal_model.joblib or signal_dataset.csv (the interference classifier).

All signals are synthetic. Accuracy numbers describe these simulated signals only.
"""
import csv
import json
import sys
import time
from collections import Counter

import numpy as np

import modulation_engine as M

TRAIN_PER_CLASS = 600
TEST_PER_CLASS = 200
UNSEEN_PER_TYPE = 150
THRESHOLD = M.DEFAULT_THRESHOLD


def make_rows(per_class, rng, split):
    rows = []
    for name in M.MOD_CLASSES:
        for _ in range(per_class):
            snr = float(rng.uniform(*M.TRAIN_SNR_RANGE))
            frame = M.generate_signal(name, int(rng.integers(1, 2 ** 31 - 1)), snr, rng)
            rows.append([split, round(snr, 2), name] + [round(float(v), 6) for v in M.extract_features(frame)])
    return rows


def main():
    started = time.time()
    print("=" * 60)
    print(" Modulation classifier (AM / FM / ASK / FSK / BPSK / QPSK)")
    print("=" * 60)
    print("[1/4] Generating synthetic signals (about 30 seconds)...")
    rows = make_rows(TRAIN_PER_CLASS, np.random.default_rng(2024), "train") + make_rows(TEST_PER_CLASS, np.random.default_rng(7), "test")
    with open(M.DATASET_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "snr_db", "label"] + M.FEATURE_NAMES)
        writer.writerows(rows)
    print("      saved", M.DATASET_PATH.name, "-", len(rows), "rows")

    print("[2/4] Training the Random Forest on the 'train' rows only...")
    X, y, Xt, yt, snr_t = M.load_dataset_csv()
    model = M.fit_model(X, y)

    print("[3/4] Evaluating on the held-out 'test' rows and on signal types the model never saw...")
    predicted = model.predict(Xt)
    accuracy = float(np.mean(predicted == yt))
    by_snr = {}
    for low, high in [(0, 5), (5, 10), (10, 20), (20, 30.01)]:
        mask = (snr_t >= low) & (snr_t < high)
        by_snr["%d-%d dB" % (low, min(high, 30))] = float(np.mean(predicted[mask] == yt[mask])) if mask.any() else None
    classes = [str(c) for c in model.classes_]
    confusion = [[int(np.sum((yt == a) & (predicted == b))) for b in classes] for a in classes]

    top = model.predict_proba(Xt).max(axis=1)
    known_accepted = float(np.mean(top >= THRESHOLD))
    rng = np.random.default_rng(99)
    unseen = {}
    for name in M.UNSEEN_TYPES:
        probs = np.array([model.predict_proba(np.array(M.extract_features(M.generate_signal(name, int(rng.integers(1, 2 ** 31 - 1)), 20.0, rng))).reshape(1, -1))[0]
                          for _ in range(UNSEEN_PER_TYPE)])
        labels = [classes[i] for i in probs.argmax(axis=1)]
        unseen[name] = {"labelled_as": dict(Counter(labels)), "mean_confidence": float(probs.max(axis=1).mean()),
                        "rejected_as_unknown": float(np.mean(probs.max(axis=1) < THRESHOLD))}

    metrics = {
        "classes": classes, "train_rows": int(len(y)), "test_rows": int(len(yt)),
        "snr_range_db": list(M.TRAIN_SNR_RANGE), "holdout_accuracy": accuracy, "accuracy_by_snr": by_snr,
        "confusion_matrix": confusion, "unknown_threshold": THRESHOLD,
        "known_frames_accepted_at_threshold": known_accepted, "unseen_types": unseen,
        "note": "Synthetic signals only. Accuracy describes simulated frames, not real radio.",
    }
    M.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    M.save_bundle(model, metrics)
    print("[4/4] Saved", M.MODEL_PATH.name, "and", M.METRICS_PATH.name)

    print("\nHold-out accuracy (%d frames, SNR 0-30 dB): %.1f%%" % (len(yt), accuracy * 100))
    for key, value in by_snr.items():
        print("   SNR %-9s %s" % (key, "n/a" if value is None else "%.1f%%" % (value * 100)))
    print("Confusion matrix (rows = true, columns = predicted):", classes)
    for name, row in zip(classes, confusion):
        print("   %-5s %s" % (name, row))
    print("Known frames above the %.0f%% limit: %.1f%%" % (THRESHOLD * 100, known_accepted * 100))
    print("Signal types the model never saw (SNR 20 dB):")
    for name, info in unseen.items():
        print("   %-22s labelled %s, mean confidence %.0f%%, rejected as UNKNOWN %.0f%%" % (
            name, info["labelled_as"], info["mean_confidence"] * 100, info["rejected_as_unknown"] * 100))
    print("\nTime: %.0f s. All data is synthetic; this is not validated on real radio signals." % (time.time() - started))


if __name__ == "__main__":
    sys.exit(main())
