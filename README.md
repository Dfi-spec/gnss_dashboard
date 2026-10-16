# GNSS Jamming Detection Analyzer - local dashboard (Python backend)

A mission-control style dashboard that runs on your Windows laptop.
The page only draws and collects button presses. **The signal, its features and the
classification are computed by a local Python (Flask) server that runs your saved
Random Forest (`signal_model.joblib`).**

> **Simulated data only.** Every signal and every satellite reading is synthetic. The dashboard is
> not connected to a GNSS receiver and has **not been validated as a real-world GNSS jamming detector.**

---------------------------------------------------------------------

## 1. Folder structure

```
gnss_dashboard\
|-- run_dashboard.bat          <- double-click to start (does the setup on first run)
|-- run_selftest.bat           <- checks that the backend really uses your model
|-- run_signal_simulator.bat   <- runs your original signal_simulator.py (graphs)
|-- retrain_model.bat          <- re-runs your ml_model.py (only if the model fails to load)
|-- retrain_modulation.bat     <- NEW: rebuilds the modulation classifier (dataset, model, metrics)
|-- setup_env.bat              <- creates .venv and installs packages (called by the others)
|-- requirements.txt           <- exact package versions
|-- server.py                  <- Flask backend + API (now also /api/link and /api/mod/*)
|-- signal_engine.py           <- signal generator + model wrapper (uses ml_model.py)
|-- link_model.py              <- NEW: simulated GNSS link (satellites, strength, C/N0, status)
|-- modulation_engine.py       <- NEW: AM/FM/ASK/FSK/BPSK/QPSK signals, features, classifier wrapper
|-- train_modulation.py        <- NEW: builds the modulation dataset, trains and tests the model
|-- modulation_model.joblib    <- NEW: trained modulation Random Forest
|-- modulation_dataset.csv     <- NEW: its dataset (train + held-out test rows, synthetic)
|-- modulation_metrics.json    <- NEW: its measured accuracy, confusion matrix, unseen-signal results
|-- static\
|   |-- index.html             <- the dashboard page (your design, plus the new parts)
|   |-- alerts.js              <- NEW: audio alert logic and sound engine
|   `-- audio\
|       `-- jammer_alert.mp3   <- NEW: the jammer alert sound (your MP3)
|-- tests\
|   |-- selftest.py            <- run by run_selftest.bat
|   |-- test_alert_gate.js     <- NEW: duplicate-alert tests (run automatically if Node.js is installed)
|   `-- browser_test.py        <- NEW, optional: clicks the real page (needs Playwright)
|-- ml_model.py                <- YOUR file, unchanged
|-- signal_model.joblib        <- YOUR trained Random Forest, unchanged
|-- signal_dataset.csv         <- YOUR dataset, unchanged
|-- signal_simulator.py        <- YOUR file, unchanged
`-- app.py                     <- YOUR Streamlit app, unchanged (not used by this dashboard)
```

## 2. Requirements

- Windows 10 or 11
- Python 3.13 from https://www.python.org/downloads/ (tick **Add python.exe to PATH**)
- Internet on the first run (to install packages, and for the Google Fonts the design uses)

Packages (installed automatically into `.venv`, versions pinned in `requirements.txt`):

| Package | Version | Used for |
|---|---|---|
| Flask | 3.1.3 | local web server |
| numpy | 2.5.4 | signals |
| scipy | 1.18.1 | FFT |
| scikit-learn | 1.9.1 | loads and runs your Random Forest (1.9.1 is the version that saved your model) |
| joblib | 1.6.0 | loads `signal_model.joblib` |
| matplotlib | 3.11.2 | only for your `signal_simulator.py` |

## 3. Run it (copy and paste into CMD)

Easiest: double-click `run_dashboard.bat`.

Or from CMD (change the path to where you unzipped the folder):

```bat
cd /d C:\Users\YourName\Downloads\gnss_dashboard
run_dashboard.bat
```

The first run creates `.venv` and installs the packages (a few minutes). Then your browser opens
http://127.0.0.1:5000. Close the CMD window (or press Ctrl+C) to stop.

Manual version, without the .bat files:

```bat
cd /d C:\Users\YourName\Downloads\gnss_dashboard
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python tests\selftest.py
python server.py
```

Then open http://127.0.0.1:5000 in Chrome or Edge.

Check the setup: `run_selftest.bat` (or `python tests\selftest.py`). It must end with `ALL CHECKS PASSED`.

Different port: `set GNSS_PORT=5001` before `run_dashboard.bat`.

## 4. What every control does

**Simulator (right unit)** - makes the simulated input
| Control | Action |
|---|---|
| SELECT | moves the `>` marker: Signal Type, Variation, Extra noise |
| UP / DOWN (hold to repeat) | changes the selected setting. Signal Type: Normal Tone, Wideband Noise, Frequency Sweep, Unfamiliar Test Signal. Variation: 1-100 (fixes frequency/strength of the signal). Extra noise: 0-100 % |
| ENTER | starts / stops the output (Status: Running / Stopped). Stopped shows NO INPUT |
| RF OUT / INTERFERENCE OUT jacks | RF OUT lights while running; INTERFERENCE OUT lights red for the noise and sweep signals |

**Analyzer (left unit)**
| Control | Action |
|---|---|
| Knob (drag, mouse wheel, or arrow keys) | UNKNOWN limit, 50-95 %. Below it the answer is UNKNOWN. Sent to the server with every request |
| FRZ | freezes the screen. Moving the knob while frozen re-judges the frozen frame through `/api/classify` |
| REF / CLR | stores / clears a reference spectrum (dashed line) |
| LOG | spectrum in dB or linear |
| GRID | grid on or off |
| WTR | spectrogram on or off |
| PEAK | marks the peak frequency |
| BAND | marks the measured bandwidth |
| AVG | averages the displayed spectrum (display only; classification uses the raw frame) |
| Left / Right arrows | simulated satellites 0-20 |
| Up / Down arrows | simulated signal strength 0-100 % |
| ENTER (nav pad) | resets the simulated link to 12 satellites, 78 % |
| POWER | screen on / off (no requests while off) |

The **GNSS LINK (SIMULATED)** panel uses the same simple rule as your `app.py`
(satellites <= 4 or strength < 25 = DEGRADED, <= 7 or < 50 = CAUTION). It is a slider-driven indicator, not
a model output and not a measurement.

## 5. How the classification is made

```
browser (static/index.html)
   |  POST /api/step  {type, seed, extra, threshold, history}
   v
server.py
   |-- signal_engine.generate_signal()      synthetic signal (same recipes as your app.py / ml_model.py)
   |-- ml_model.extract_features()          peak frequency, bandwidth, spectral flatness, RMS
   |-- ml_model.classify_signal()           -> signal_model.joblib  (predict_proba)
   |-- smoothing over the last 4 frames + UNKNOWN rule (top probability < threshold)
   v
JSON {samples, spectrum, features, probabilities, prediction, status}
   v
browser only draws it
```

Status colours: green = Normal Tone, red = Wideband Noise or Frequency Sweep (shown as "jammer"), amber = UNKNOWN.

### API (all local)
- `GET  /api/status` - model facts, classes, dataset facts, hold-out accuracy from `signal_dataset.csv`
- `POST /api/step` - body `{"type":0-3,"seed":1-100,"extra":0-100,"threshold":0.5-0.95,"history":[...]}`
- `POST /api/classify` - body `{"features":[peak,bandwidth,flatness,rms],"threshold":0.8}`

## 6. Things to know (please read before presenting)

- **Unknown limit.** `ml_model.py` saved 70 %. The Unfamiliar Test Signal gets about 69-71 % from your forest, right at that limit,
  so the dashboard starts at 80 %, where it is reliably UNKNOWN. To start at 70 % like `ml_model.py`:
  `set GNSS_UNKNOWN_THRESHOLD=0.70` before `run_dashboard.bat`.
- **UNKNOWN is a confidence rule,** not proof that a signal is new. A different unfamiliar signal can still get a high confidence.
- **100 % accuracy** (the hold-out check in the footer, computed on `signal_dataset.csv`) is because your three known signals look very different.
  It says nothing about real radio signals.
- **Normal Tone** is treated as the "normal" class and the other two as "jammer" classes. That is a labelling choice of this demo.
- A frame that is not a clean example of the training signals (for example a Normal Tone with a lot of extra noise) can be misclassified. That is a real limit of the model.
- This package was tested on Linux with the exact package versions above, using a Flask test client and a scripted browser-like page test.
  It has not been run on a Windows machine, so please run `run_selftest.bat` first.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| "Python was not found" | Install Python 3.13, tick **Add python.exe to PATH**, run the .bat again |
| Package install fails | Check the internet connection, run the .bat again (it resumes) |
| "Could not load signal_model.joblib" | The installed scikit-learn differs from the one that saved the model. Delete the `.venv` folder and run again (it installs 1.9.1). If it still fails, run `retrain_model.bat` |
| Page says BACKEND OFFLINE | The CMD window with the server was closed. Run `run_dashboard.bat` and reload |
| Port 5000 is in use | `set GNSS_PORT=5001` then `run_dashboard.bat` |
| Fonts look different | The design loads Google Fonts. Offline it falls back to default fonts |
| Windows Firewall prompt | The server listens on 127.0.0.1 only (your own laptop). You can click Cancel |


---------------------------------------------------------------------

## 8. What is new in this version

**1. GNSS LINK panel is now live (simulated).** It used to be two sliders. Now `link_model.py` (server side)
computes it from the simulator: signal type, Extra noise and Running/Stopped. The nav pad sets the clear-sky link
(satellites in view, strength). Interference removes satellites and lowers strength/C/N0 gradually; Stopped shows NO SIGNAL.
Same DEGRADED/CAUTION thresholds as before. It is an illustrative model, not a measurement.

**2. MODULATION CLASSIFIER (new third unit, separate from the jamming analyzer).** Own model
(`modulation_model.joblib`), own data (`modulation_dataset.csv`), own metrics (`modulation_metrics.json`).
Trained classes: AM, FM (analog) and ASK, FSK, BPSK, QPSK (digital). Controls on its LCD: SELECT / UP / DOWN / ENTER
(Source, Variation, SNR, Reject limit, Status), FRZ, LOG, POWER. Rebuild it with `retrain_modulation.bat`.
If the saved model was made by a different scikit-learn version, the server rebuilds it from the CSV automatically on start.

**3. AUDIO ALERTS (optional, OFF until you press SOUND).** Group "ALERT AUDIO" on the analyzer: SOUND on/off, MUTE (also cuts off
a sound that is playing), TEST (plays the real alert sound at the current volume), VOL fader. Alerts: JAMMER, UNKNOWN SIGNAL, LINK DEGRADED
(they apply to the jamming analyzer and its link, not to the modulation unit).
- **The JAMMER alert plays your MP3**: `static/audio/jammer_alert.mp3` (1.6 s). To use another sound, replace that file and keep the name.
  It is downloaded when the page opens and decoded on your first click, silence at the very start of the file is skipped, so it starts
  immediately. If the file is missing, simple beeps are used instead. UNKNOWN / LINK DEGRADED still use short beeps.
- **Instant:** the alert does not wait for the screen. The screen smooths the last 4 frames (stable, but it needs about 0.6 s to settle on a new
  signal). The alert uses the single-frame decision from the server (`instant` in `/api/step`, same 80% rule, no history) and judges each frame the
  moment it arrives. A jammer sounds on the first frame after the change; the badge on the screen follows about half a second later.
- **Duplicate prevention:** a jammer needs 1 frame, UNKNOWN 2 (so the brief UNKNOWN while one signal changes into another stays silent),
  LINK DEGRADED 5; one sound per event; it re-arms only after the state has cleared (3 normal frames); 8 s cool-down per kind; priority
  jammer > unknown > degraded; two sounds never overlap. Sounds play only on a new frame (not while frozen, not while the analyzer is off).
- Browsers only allow sound after a click, which is why SOUND (or TEST) must be pressed once.

### Honest limits of the modulation classifier (synthetic signals only)
- Hold-out accuracy 99.7% on simulated frames (SNR 0-30 dB). That says nothing about real radio signals.
- At 0 dB SNR a single frame is often UNKNOWN (about 30%) instead of wrong. Confident wrong labels are rare.
- UNKNOWN is a confidence rule. "Noise only" is rejected; **16-QAM is mostly rejected but sometimes labelled QPSK; a Chirp is labelled FM
  (a chirp is a linear FM) and is NOT rejected.** Sources marked * on the LCD were never in the training set.

### New tests
- `run_selftest.bat` now also checks the link model, the modulation classifier, the instant alert decision and the MP3 file, that your interference
  model/data/code files are byte-identical to the originals (section 10), and (if Node.js is installed) the alert logic. It must end with `ALL CHECKS PASSED`.
- `tests\browser_test.py` is OPTIONAL (needs Playwright). It clicks the real page, counts every sound with a fake audio object, decodes the real MP3
  in a real browser and checks that the jammer alert fires on the first frame after a change.
- The live signals use fresh random noise. Measured on 20,000 frames each: a Frequency Sweep frame is right but at only 74-77% confidence (so the 80% rule
  says UNKNOWN) about 0.015% of the time, Wideband Noise 0.005%, Normal Tone 0%, and a WRONG class never happened. So the "always classified" self-test
  checks tolerate one such borderline frame in 60 and still fail on any wrong class. If a check fails once, run `run_selftest.bat` again.
- Not run on Windows: the `.bat` files were checked by reading and the same steps were run on Linux.
