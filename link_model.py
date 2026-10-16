"""
link_model.py - SIMULATED GNSS link health for the dashboard's "GNSS LINK" panel.

Before this module the panel was only two sliders (satellites, strength). It did not
react to the signal. Now it is computed here, on the server, from:

  * the nav-pad settings            -> the "clear-sky" link (satellites in view, strength %)
  * the simulator's signal type     -> how much interference reaches the receiver
  * the simulator's "Extra noise"   -> adds to the interference
  * the simulator's Running/Stopped -> Stopped means no GNSS signal at all

The model is deliberately simple and is ILLUSTRATIVE ONLY. It is not a measurement and it
has not been validated against a real GNSS receiver:

  clear-sky C/N0   = 15 + 0.35 * strength%                      (78 % -> about 42 dB-Hz)
  C/N0 with jammer = clear-sky C/N0 - 25 dB * interference      (interference 0..1)
  each satellite i has its own offset (+5 dB ... -8 dB, like different elevations) and is
  "tracked" while its C/N0 stays at or above 25 dB-Hz
  strength%        = (C/N0 - 15) / 0.35

So stronger interference first removes the weaker satellites, then the rest. The status
uses the same thresholds as the original app.py (<=4 sats or <25 %  -> DEGRADED,
<=7 sats or <50 % -> CAUTION).
"""
import numpy as np

MAX_SATS = 20
TRACK_THRESHOLD_DBHZ = 25.0     # a satellite is lost below this C/N0
CNO_FLOOR = 15.0                # dB-Hz at 0 % strength
CNO_PER_PERCENT = 0.35          # dB-Hz per 1 % strength
MAX_DROP_DB = 25.0              # C/N0 lost at interference = 1.0
SMOOTHING = 0.35                # share of the gap closed per frame (links change gradually)
OFFSET_TOP_DB, OFFSET_BOTTOM_DB = 5.0, -8.0

# interference index (0 = none ... 1 = total) the simulated source puts on the GNSS band
BASE_INTERFERENCE = {
    0: 0.00,   # Normal Tone: clean GNSS-like input
    1: 0.80,   # Wideband Noise (raised further by louder noise, see interference_index)
    2: 0.55,   # Frequency Sweep
    3: 0.35,   # Unfamiliar Test Signal (two tones + noise)
}
EXTRA_NOISE_WEIGHT = 0.60       # 100 % extra noise adds 0.60


def clamp(value, low, high):
    return max(low, min(high, value))


def interference_index(type_index, extra_percent, rms=None):
    """0..1 amount of interference caused by the simulated source and its extra noise."""
    base = BASE_INTERFERENCE.get(int(type_index), 0.0)
    if int(type_index) == 1 and rms is not None:
        # louder wideband noise (training range sigma 0.3-0.8) hurts more: 0.65 .. 1.00
        base = 0.65 + 0.35 * clamp((float(rms) - 0.3) / 0.5, 0.0, 1.0)
    index = base + EXTRA_NOISE_WEIGHT * clamp(float(extra_percent), 0.0, 100.0) / 100.0
    return float(clamp(index, 0.0, 1.0))


def clear_sky_cno(strength_percent):
    return CNO_FLOOR + CNO_PER_PERCENT * clamp(float(strength_percent), 0.0, 100.0)


def status_for(tracked, strength):
    """Same thresholds as the original app.py / dashboard."""
    if tracked <= 4 or strength < 25:
        return "DEGRADED"
    if tracked <= 7 or strength < 50:
        return "CAUTION"
    return "NOMINAL"


def compute_link(interference, running, sats_in_view=12, strength_percent=78, prev_cno=None, rng=None):
    """
    Returns a dict describing the simulated link.

    interference      0..1 from interference_index()
    running           False = the simulator output is stopped -> NO SIGNAL
    sats_in_view      0..20 from the nav pad (satellites above the horizon in the simulation)
    strength_percent  0..100 from the nav pad (clear-sky signal strength)
    prev_cno          the "cno_trend" returned by the previous call (gives gradual changes)
    rng               numpy Generator for the small live fluctuation of the C/N0 readout
    """
    sats_in_view = int(clamp(int(sats_in_view), 0, MAX_SATS))
    strength_percent = int(clamp(int(round(float(strength_percent))), 0, 100))
    interference = float(clamp(float(interference), 0.0, 1.0))

    base = {
        "sats_in_view": sats_in_view,
        "baseline_strength": strength_percent,
        "interference": round(interference, 3),
    }
    if not running:
        base.update({"sats_tracked": 0, "strength": 0, "cno_dbhz": None, "cno_trend": None,
                     "status": "NO SIGNAL", "signal_present": False})
        return base

    target = clear_sky_cno(strength_percent) - MAX_DROP_DB * interference
    target = float(clamp(target, 0.0, 55.0))
    trend = target if prev_cno is None else float(prev_cno) + SMOOTHING * (target - float(prev_cno))
    trend = float(clamp(trend, 0.0, 55.0))

    if sats_in_view > 1:
        offsets = np.linspace(OFFSET_TOP_DB, OFFSET_BOTTOM_DB, sats_in_view)
    else:
        offsets = np.zeros(sats_in_view)
    tracked = int(np.sum(trend + offsets >= TRACK_THRESHOLD_DBHZ))
    strength = int(round(clamp((trend - CNO_FLOOR) / CNO_PER_PERCENT, 0.0, 100.0)))

    # the readout wobbles a little, more under interference; the status never uses the wobble
    rng = rng if rng is not None else np.random.default_rng()
    wobble = rng.normal(0.0, 0.15 + 0.5 * interference)
    cno_shown = float(clamp(trend + wobble, 0.0, 55.0))

    base.update({
        "sats_tracked": tracked,
        "strength": strength,
        "cno_dbhz": round(cno_shown, 1),
        "cno_trend": round(trend, 3),
        "status": status_for(tracked, strength),
        "signal_present": True,
    })
    return base
