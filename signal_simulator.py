
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

sample_rate = 1000
duration = 1
t = np.arange(0, duration, 1 / sample_rate)

signals = {
    "Normal Tone": np.sin(2 * np.pi * 50 * t),
    "Wideband Noise": np.random.normal(0, 0.5, len(t)),
    "Frequency Sweep": np.sin(2 * np.pi * (10 * t + 40 * t**2))
}

def extract_features(signal):
    spectrum = np.abs(rfft(signal))
    frequencies = rfftfreq(len(signal), 1 / sample_rate)

    peak_frequency = frequencies[np.argmax(spectrum)]

    # Estimate bandwidth using frequencies containing
    # at least 10% of the maximum spectrum magnitude.
    threshold = 0.1 * np.max(spectrum)
    active_frequencies = frequencies[spectrum >= threshold]

    if len(active_frequencies) > 0:
        bandwidth = active_frequencies[-1] - active_frequencies[0]
    else:
        bandwidth = 0

    # Spectral flatness: geometric mean / arithmetic mean
    power = spectrum**2 + 1e-12
    spectral_flatness = (
        np.exp(np.mean(np.log(power))) / np.mean(power)
    )

    return peak_frequency, bandwidth, spectral_flatness


print("\nSIGNAL FEATURE ANALYSIS")
print("-" * 65)

fig, axes = plt.subplots(3, 2, figsize=(12, 9))

for row, (name, signal) in enumerate(signals.items()):
    peak, bandwidth, flatness = extract_features(signal)

    print(f"\n{name}")
    print(f"  Peak frequency:    {peak:.2f} Hz")
    print(f"  Bandwidth estimate: {bandwidth:.2f} Hz")
    print(f"  Spectral flatness:  {flatness:.4f}")

    axes[row, 0].plot(t, signal)
    axes[row, 0].set_title(f"{name} - Time Domain")
    axes[row, 0].set_xlabel("Time (seconds)")
    axes[row, 0].set_ylabel("Amplitude")

    spectrum = np.abs(rfft(signal))
    frequencies = rfftfreq(len(signal), 1 / sample_rate)

    axes[row, 1].plot(frequencies, spectrum)
    axes[row, 1].set_title(f"{name} - FFT Spectrum")
    axes[row, 1].set_xlabel("Frequency (Hz)")
    axes[row, 1].set_ylabel("Magnitude")
    axes[row, 1].set_xlim(0, 250)

plt.tight_layout()
plt.show()
