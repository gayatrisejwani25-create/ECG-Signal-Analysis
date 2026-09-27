"""
ECG AS A SIGNAL — Signals & Systems Project
=============================================
An interactive Streamlit app that walks through the complete pipeline of
turning the heart's continuous electrical activity x(t) into a digital
signal x[n], and back into insight:

    Heart -> ECG Sensor -> Continuous Signal x(t) -> Sampling -> Digital
    ECG x[n] -> Filtering -> R-Peak Detection -> Heart Rate

Features
--------
1. Theory walkthrough (continuous vs discrete signals, sampling theorem,
   acquisition chain, applications).
2. FIVE audible + visible heartbeat examples, chosen through a simple
   step-by-step flow (Choose -> Listen & Watch -> Analyze -> Compare):
     - Normal
     - Tachycardia (High Rate)
     - Bradycardia (Low Rate)
     - Irregular (Arrhythmia-like)
     - Cardiac Arrest / Flatline (Asystole) — demonstrates what the ECG
       trace AND the monitor alarm sound like when a patient's heart
       stops, so the "no heartbeat" case is covered end-to-end too.
3. A real ECG recording (ecg_sample.csv) used to demonstrate sampling,
   digital band-pass filtering and R-peak / heart-rate detection.
4. A side-by-side comparison of all five heartbeat types against the
   normal reference range, so you can see how each one deviates from
   "how a heartbeat should look" — including the flatline case, which
   deviates from all of them by having no heartbeat at all.

Educational project — NOT intended for medical diagnosis.
"""

import io
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io.wavfile import write as write_wav
from scipy.signal import butter, filtfilt, find_peaks
from streamlit.components.v1 import html as components_html

# =====================================================================
# PAGE CONFIG + STYLE
# =====================================================================

st.set_page_config(
    page_title="ECG as a Signal | Signals & Systems",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main > div {padding-top: 1rem;}
    .hero {
        background: linear-gradient(120deg, #7f1d1d 0%, #b91c1c 45%, #ef4444 100%);
        padding: 2rem 2.2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 30px rgba(185,28,28,0.25);
    }
    .hero h1 {margin: 0; font-size: 2.1rem;}
    .hero p {margin: 0.35rem 0 0 0; opacity: 0.92; font-size: 1.02rem;}
    .flow-badge {
        display: inline-block; background: rgba(255,255,255,0.16);
        padding: 0.3rem 0.8rem; border-radius: 999px; margin: 0.15rem 0.2rem 0 0;
        font-size: 0.82rem; font-weight: 600; letter-spacing: 0.2px;
    }
    .card {
        background: var(--background-color, #111827);
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
    }
    .step-badge {
        display:inline-block; background:#1f2937; color:#e5e7eb;
        border-radius:999px; width:26px; height:26px; text-align:center;
        line-height:26px; font-weight:700; margin-right:8px; font-size:0.85rem;
    }
    .metric-good {color: #22c55e; font-weight: 700;}
    .metric-warn {color: #f59e0b; font-weight: 700;}
    .metric-bad {color: #ef4444; font-weight: 700;}
    .flatline-alert {
        background: #450a0a; border: 2px solid #ef4444; border-radius: 14px;
        padding: 1rem 1.3rem; color: #fecaca; font-weight: 600;
        animation: pulseAlert 1.1s infinite;
    }
    @keyframes pulseAlert {
        0%   {box-shadow: 0 0 0 0 rgba(239,68,68,0.55);}
        70%  {box-shadow: 0 0 0 14px rgba(239,68,68,0);}
        100% {box-shadow: 0 0 0 0 rgba(239,68,68,0);}
    }
    section[data-testid="stSidebar"] {border-right: 1px solid rgba(148,163,184,0.2);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>❤️ ECG as a Signal</h1>
        <p>Modeling cardiac electrical activity — a Signals &amp; Systems project</p>
        <div style="margin-top:0.8rem;">
            <span class="flow-badge">Heart</span> →
            <span class="flow-badge">ECG Sensor</span> →
            <span class="flow-badge">Continuous x(t)</span> →
            <span class="flow-badge">Sampling</span> →
            <span class="flow-badge">Digital x[n]</span> →
            <span class="flow-badge">Filtering</span> →
            <span class="flow-badge">R-Peak Detection</span> →
            <span class="flow-badge">Heart Rate</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# CONSTANTS
# =====================================================================

AUDIO_FS = 44100  # audio sample rate for playback (Hz)

# A simple 5-Gaussian "textbook" ECG beat model (P, Q, R, S, T waves).
# (name, amplitude, center as a fraction of a nominal 0.8 s beat, width in s)
WAVES = [
    ("P", 0.15, 0.24, 0.028),
    ("Q", -0.12, 0.40, 0.010),
    ("R", 1.20, 0.44, 0.009),
    ("S", -0.28, 0.48, 0.010),
    ("T", 0.35, 0.68, 0.045),
]
NOMINAL_RR = 0.8  # seconds, the beat duration WAVES was designed around

FLATLINE_DURATION_S = 12.0  # how long the flatline example plays for

# NOTE: dict order below defines the order shown throughout the app
# (selector, live monitor, comparison table, comparison charts).
HEARTBEAT_TYPES = {
    "Normal Heartbeat": {
        "icon": "💚",
        "color": "#22c55e",
        "base_rr": 0.90,
        "jitter": 0.015,
        "range": (0.80, 1.00),
        "expected_bpm": (60, 100),
        "flatline": False,
        "description": (
            "A healthy resting rhythm: roughly evenly spaced beats, "
            "heart rate settling between 60 and 100 BPM."
        ),
    },
    "Tachycardia (High Rate)": {
        "icon": "🔴",
        "color": "#ef4444",
        "base_rr": 0.45,
        "jitter": 0.010,
        "range": (0.40, 0.50),
        "expected_bpm": (100, 160),
        "flatline": False,
        "description": (
            "An abnormally fast heart rate (>100 BPM) — beats arrive in "
            "quick succession, shortening the RR interval."
        ),
    },
    "Bradycardia (Low Rate)": {
        "icon": "🔵",
        "color": "#3b82f6",
        "base_rr": 1.45,
        "jitter": 0.020,
        "range": (1.35, 1.55),
        "expected_bpm": (30, 55),
        "flatline": False,
        "description": (
            "An abnormally slow heart rate (<60 BPM) — long, stretched-out "
            "gaps appear between beats."
        ),
    },
    "Irregular (Arrhythmia-like)": {
        "icon": "🟠",
        "color": "#f59e0b",
        "base_rr": None,  # handled specially: highly variable RR
        "jitter": None,
        "range": (0.45, 1.55),
        "expected_bpm": None,
        "flatline": False,
        "description": (
            "An irregular rhythm — RR intervals swing unpredictably, "
            "similar in spirit to an arrhythmia such as AFib."
        ),
    },
    "Cardiac Arrest (Flatline / Asystole)": {
        "icon": "⚫",
        "color": "#9ca3af",
        "base_rr": None,
        "jitter": None,
        "range": (0.0, 0.0),
        "expected_bpm": (0, 0),
        "flatline": True,
        "description": (
            "No electrical activity — **asystole**. The trace goes flat and "
            "the monitor switches from beat-tones to a single, continuous "
            "alarm tone. This is the classic 'patient has died / cardiac "
            "arrest' signal you hear in hospital dramas — modeled here for "
            "educational comparison against a real heartbeat."
        ),
    },
}


def is_flatline(kind: str) -> bool:
    return HEARTBEAT_TYPES[kind]["flatline"]


# =====================================================================
# DATA LOADING (the real, uploaded ECG recording)
# =====================================================================


@st.cache_data
def load_ecg_csv(path: str = "ecg_sample.csv") -> np.ndarray:
    """Robustly load the single-column ECG recording shipped with the app."""
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    col = df.columns[0]
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col])
    return df[col].to_numpy()


# =====================================================================
# SIGNAL PROCESSING HELPERS
# =====================================================================


def bandpass_filter(signal: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0) -> np.ndarray:
    """4th-order Butterworth band-pass filter, matched to typical ECG content."""
    nyquist = fs / 2
    high = min(high, nyquist * 0.98)
    if high <= low:
        return signal
    b, a = butter(4, [low / nyquist, high / nyquist], btype="bandpass")
    return filtfilt(b, a, signal)


def detect_r_peaks(signal: np.ndarray, fs: float, mode: str = "real") -> np.ndarray:
    """
    Detect R-peaks.
    mode="real"  -> tuned for arbitrary real-world recordings (std-based).
    mode="synth" -> tuned for the clean synthetic beat model (amplitude-based).

    For a flatline signal, callers should skip this function entirely
    (see `analyze_kind` below) rather than rely on it to "discover" that
    there are no peaks — a near-zero noise floor can otherwise still
    produce spurious low-amplitude "peaks" relative to itself.
    """
    distance = int(0.25 * fs)
    if mode == "synth":
        ref = np.max(np.abs(signal)) or 1.0
        peaks, _ = find_peaks(signal, distance=distance, prominence=0.4 * ref, height=0.35 * ref)
    else:
        prominence = 0.5 * np.std(signal)
        peaks, _ = find_peaks(signal, distance=max(distance, int(0.35 * fs)), prominence=prominence)
    return peaks


def heart_rate_metrics(peaks: np.ndarray, fs: float):
    """Return (rr_intervals_seconds, heart_rate_bpm, sdnn_ms) or (None, None, None)."""
    if peaks is None or len(peaks) < 2:
        return None, None, None
    rr = np.diff(peaks) / fs
    hr = 60.0 / np.mean(rr)
    sdnn = np.std(rr) * 1000.0
    return rr, hr, sdnn


def verdict_for(hr, sdnn):
    """A short, friendly interpretation of measured heart-rate stats."""
    if hr is None:
        return "Not enough beats detected", "metric-bad"
    if sdnn is not None and sdnn > 120:
        return "Irregular rhythm (high beat-to-beat variability)", "metric-warn"
    if hr < 60:
        return "Bradycardia (slower than normal resting range)", "metric-warn"
    if hr > 100:
        return "Tachycardia (faster than normal resting range)", "metric-warn"
    return "Within normal resting range (60–100 BPM)", "metric-good"


# ---------------------------------------------------------------------
# Synthetic multi-beat ECG generator (used for the 5 heartbeat examples)
# ---------------------------------------------------------------------


def synth_beat(rr: float, fs: float) -> np.ndarray:
    """Generate one P-Q-R-S-T beat lasting `rr` seconds at sample rate fs."""
    n = max(20, int(round(rr * fs)))
    t = np.linspace(0, rr, n, endpoint=False)
    scale = rr / NOMINAL_RR
    y = np.zeros(n)
    for _name, amp, center_frac, width in WAVES:
        center = center_frac * NOMINAL_RR * scale
        y += amp * np.exp(-((t - center) ** 2) / (2 * (width * max(scale, 0.6)) ** 2))
    return y


def rr_sequence(kind: str, n_beats: int = 14, seed: int = 7) -> list:
    """Generate a ground-truth list of RR intervals (seconds) for a heartbeat type.
    Returns an empty list for the flatline type — there are no beats."""
    if is_flatline(kind):
        return []
    rng = np.random.default_rng(seed)
    cfg = HEARTBEAT_TYPES[kind]
    if kind.startswith("Irregular"):
        low, high = cfg["range"]
        return list(rng.uniform(low, high, n_beats))
    low, high = cfg["range"]
    values = cfg["base_rr"] + rng.normal(0, cfg["jitter"], n_beats)
    return list(np.clip(values, low, high))


def build_ecg_train(rr_intervals: list, fs: float) -> np.ndarray:
    """Concatenate synthetic beats back-to-back to form a multi-beat signal."""
    return np.concatenate([synth_beat(rr, fs) for rr in rr_intervals])


def build_flatline_signal(fs: float, duration_s: float = FLATLINE_DURATION_S, seed: int = 7) -> np.ndarray:
    """A near-zero, essentially flat trace — asystole. Tiny sensor noise is
    kept so it still looks like a real (if lifeless) recording rather than
    a perfect mathematical zero line."""
    n = max(20, int(round(duration_s * fs)))
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.0015, n)


def build_signal_for_kind(kind: str, fs: float, n_beats: int = 40, seed: int = 7):
    """
    Unified signal builder used everywhere in the app.
    Returns (signal, rr_list) where rr_list is None for the flatline type.
    """
    if is_flatline(kind):
        return build_flatline_signal(fs, seed=seed), None
    rr_list = rr_sequence(kind, n_beats=n_beats, seed=seed)
    return build_ecg_train(rr_list, fs), rr_list


def analyze_kind(kind: str, signal: np.ndarray, fs: float):
    """
    Run the full sampling -> filtering -> R-peak detection -> heart-rate
    pipeline for a given heartbeat type's signal, with an explicit
    flatline short-circuit (asystole never goes through peak detection —
    it is reported directly as "no heartbeat").
    Returns (filtered_signal, peaks, rr, hr, sdnn, verdict, css_class).
    """
    filtered = bandpass_filter(signal, fs)
    if is_flatline(kind):
        peaks = np.array([], dtype=int)
        rr, hr, sdnn = None, None, None
        verdict = "🚨 Asystole — no cardiac electrical activity detected"
        css_class = "metric-bad"
        return filtered, peaks, rr, hr, sdnn, verdict, css_class

    peaks = detect_r_peaks(filtered, fs, mode="synth")
    rr, hr, sdnn = heart_rate_metrics(peaks, fs)
    verdict, css_class = verdict_for(hr, sdnn)
    return filtered, peaks, rr, hr, sdnn, verdict, css_class


def render_live_ecg_monitor(kind: str, signal: np.ndarray, columns_per_second: int = 150, height: int = 260):
    """
    Render a genuinely MOVING, continuously-sweeping ECG monitor — like a
    real bedside cardiac monitor — using an HTML5 <canvas> animated with
    requestAnimationFrame (runs entirely in the browser, so it never
    freezes or "finishes" the way a Python-side loop does).

    The trace is drawn as a moving pen: new samples are written just ahead
    of a small blank gap, and once the sweep reaches the right edge it
    wraps back to the left and starts overwriting the old trace — exactly
    like a real ECG / vitals monitor. For the flatline type, the pen draws
    a flat, unbroken red line and the header switches to a pulsing
    "ASYSTOLE" alarm label, matching how a real bedside monitor behaves
    the moment it loses cardiac activity.
    """
    cfg = HEARTBEAT_TYPES[kind]
    flat = is_flatline(kind)

    amp = np.max(np.abs(signal)) or 1.0
    normalized = (signal / amp).tolist() if not flat else (signal * 0.0).tolist()
    data_json = json.dumps(normalized)

    canvas_id = f"ecgCanvas_{abs(hash(kind)) % 100000}"
    trace_color = "#ef4444" if flat else cfg["color"]
    label = "⚠ ASYSTOLE — NO PULSE" if flat else kind.upper()
    label_color = "#ef4444" if flat else cfg["color"]
    pulse_css = (
        "animation: monitorPulse 0.9s infinite;" if flat else ""
    )

    template = """
    <div style="background:#020617;border-radius:14px;padding:10px 14px;
                border:1px solid #1f2937;">
      <style>
        @keyframes monitorPulse { 0%{opacity:1;} 50%{opacity:0.35;} 100%{opacity:1;} }
      </style>
      <div style="display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:6px;">
        <span style="color:#9ca3af;font:600 12px sans-serif;letter-spacing:.6px;">
          &#128137; LIVE ECG MONITOR
        </span>
        <span style="color:__LABEL_COLOR__;font:700 13px sans-serif;__PULSE_CSS__">__LABEL__</span>
      </div>
      <canvas id="__CANVAS_ID__" width="900" height="__CANVAS_H__"
              style="width:100%;display:block;border-radius:8px;background:#000;">
      </canvas>
    </div>
    <script>
    (function () {
        const data = __DATA__;
        const color = "__COLOR__";
        const canvas = document.getElementById("__CANVAS_ID__");
        const ctx = canvas.getContext("2d");
        const W = canvas.width, H = canvas.height;
        const screen = new Array(W).fill(null);
        const pxPerSecond = __PXPS__;   // columns advanced per real second
        const gapAhead = __GAP__;       // blank "pen tip" gap ahead of the trace

        let n = 0;            // total columns written so far
        let pointer = 0;       // fractional accumulator
        let lastTime = null;

        function drawGrid() {
            ctx.fillStyle = "#000000";
            ctx.fillRect(0, 0, W, H);
            ctx.strokeStyle = "rgba(31,58,42,0.9)";
            ctx.lineWidth = 1;
            for (let x = 0; x < W; x += 20) {
                ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
            }
            for (let y = 0; y < H; y += 20) {
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
            }
        }

        function step(ts) {
            if (lastTime === null) lastTime = ts;
            const dt = (ts - lastTime) / 1000;
            lastTime = ts;

            pointer += pxPerSecond * dt;

            while (pointer >= 1) {
                pointer -= 1;
                const dataIdx = n % data.length;
                const col = n % W;
                screen[col] = data[dataIdx];
                for (let g = 1; g <= gapAhead; g++) {
                    screen[(col + g) % W] = null;
                }
                n++;
            }

            drawGrid();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            let started = false;
            for (let x = 0; x < W; x++) {
                const v = screen[x];
                if (v === null || v === undefined) { started = false; continue; }
                const y = H / 2 - v * (H * 0.42);
                if (!started) { ctx.moveTo(x, y); started = true; }
                else { ctx.lineTo(x, y); }
            }
            ctx.stroke();

            requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
    })();
    </script>
    """

    html_code = (
        template.replace("__DATA__", data_json)
        .replace("__PXPS__", str(columns_per_second))
        .replace("__GAP__", "0" if flat else "14")
        .replace("__COLOR__", trace_color)
        .replace("__LABEL__", label)
        .replace("__LABEL_COLOR__", label_color)
        .replace("__PULSE_CSS__", pulse_css)
        .replace("__CANVAS_ID__", canvas_id)
        .replace("__CANVAS_H__", str(height))
    )
    components_html(html_code, height=height + 60)


def synthesize_heartbeat_audio(kind: str, rr_intervals=None, sample_rate: int = AUDIO_FS,
                                flatline_duration_s: float = FLATLINE_DURATION_S) -> bytes:
    """
    Turn a heartbeat type into audio.
    - Normal / Tachycardia / Bradycardia / Irregular -> a "lub-dub" S1/S2
      heart-sound sequence timed to the given RR intervals.
    - Cardiac Arrest (Flatline) -> a single, continuous monitor alarm
      tone (the classic "flatline beep" heard when a patient's heart
      stops), instead of any lub-dub beats.
    """
    if is_flatline(kind):
        return synthesize_flatline_alarm(flatline_duration_s, sample_rate)

    total_duration = float(np.sum(rr_intervals)) + 1.0
    t = np.linspace(0, total_duration, int(sample_rate * total_duration), endpoint=False)
    audio = np.zeros_like(t)
    starts = np.concatenate(([0.0], np.cumsum(rr_intervals)[:-1]))

    for start in starts:
        idx = t >= start
        local_t = t[idx] - start

        # S1 ("lub")
        env1 = np.exp(-35 * local_t)
        lub = np.sin(2 * np.pi * 70 * local_t) * env1

        # S2 ("dub"), slightly delayed and softer
        dub_delay = 0.15
        env2 = np.zeros_like(local_t)
        after = local_t >= dub_delay
        env2[after] = np.exp(-40 * (local_t[after] - dub_delay))
        dub = np.sin(2 * np.pi * 110 * local_t) * env2

        audio[idx] += 0.85 * lub + 0.55 * dub

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    audio_i16 = (audio * 32767 * 0.9).astype(np.int16)

    buffer = io.BytesIO()
    write_wav(buffer, sample_rate, audio_i16)
    return buffer.getvalue()


def synthesize_flatline_alarm(duration_s: float = FLATLINE_DURATION_S, sample_rate: int = AUDIO_FS) -> bytes:
    """
    The classic hospital-monitor 'flatline' alarm: a single, sustained,
    unbroken tone (real bedside monitors hold this continuously until a
    clinician silences/resets it). Modeled here as a steady ~900 Hz tone
    with a very short fade-in/out to avoid a click at the edges.
    """
    n = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    freq = 900.0  # Hz — a typical continuous cardiac-monitor alarm pitch
    tone = np.sin(2 * np.pi * freq * t)

    fade_n = max(1, int(0.01 * sample_rate))
    env = np.ones(n)
    env[:fade_n] = np.linspace(0, 1, fade_n)
    env[-fade_n:] = np.linspace(1, 0, fade_n)

    audio = tone * env * 0.55
    audio_i16 = (audio * 32767).astype(np.int16)

    buffer = io.BytesIO()
    write_wav(buffer, sample_rate, audio_i16)
    return buffer.getvalue()


# =====================================================================
# LOAD REAL DATA
# =====================================================================

try:
    real_ecg = load_ecg_csv("ecg_sample.csv")
    csv_ok = True
except Exception as exc:  # noqa: BLE001
    csv_ok = False
    real_ecg = None
    csv_error = exc

# =====================================================================
# SIDEBAR
# =====================================================================

with st.sidebar:
    st.header("⚙️ Controls")
    st.caption(
        "Choose which heartbeat to listen to, watch and analyze from the "
        "**🔊 Choose, Listen & Analyze** tab. This panel only holds settings "
        "for the *real* recorded ECG demo."
    )

    st.divider()
    st.subheader("Real ECG acquisition")
    fs = st.number_input(
        "Sampling frequency of the recording (Hz)",
        min_value=50,
        max_value=2000,
        value=360,
        step=10,
        help="The rate at which the uploaded ECG recording was originally sampled.",
    )

    st.divider()
    st.caption(
        "📚 Educational project for a Signals & Systems course. "
        "Not intended for medical diagnosis."
    )

# =====================================================================
# TABS
# =====================================================================

tab_theory, tab_listen, tab_sampling, tab_rpeak, tab_compare = st.tabs(
    [
        "📖 Theory",
        "🔊 Choose, Listen & Analyze",
        "🧮 Sampling & Filtering",
        "📈 R-Peaks & Heart Rate",
        "⚖️ Compare All 5 Types",
    ]
)

# ---------------------------------------------------------------------
# TAB 1 — THEORY
# ---------------------------------------------------------------------
with tab_theory:
    st.subheader("What is an ECG?")
    st.markdown(
        """
An **electrocardiogram (ECG / EKG)** is a recording of the heart's electrical
activity over time. Every heartbeat begins with a small electrical impulse
that spreads through the cardiac muscle and triggers contraction. Because the
heart's voltage varies smoothly from one instant to the next, it is
**naturally a continuous-time signal**, captured at the skin's surface by
electrodes as tiny (millivolt-range) voltage changes:

$$x(t) = \\text{ECG amplitude (mV) at time } t$$

A single heartbeat traces out the classic **P-Q-R-S-T** waveform:
"""
    )
    fig_pqrst, ax_pqrst = plt.subplots(figsize=(9, 2.6))
    demo_beat = synth_beat(0.8, 1000)
    demo_t = np.linspace(0, 0.8, len(demo_beat))
    ax_pqrst.plot(demo_t, demo_beat, color="#ef4444", linewidth=2)
    for name, amp, center_frac, _w in WAVES:
        cx = center_frac * 0.8
        cy = amp if abs(amp) > 0.05 else amp + 0.05
        ax_pqrst.annotate(name, (cx, cy), textcoords="offset points", xytext=(0, 8 if amp > 0 else -14),
                           ha="center", fontweight="bold", color="#111827" if amp > 0 else "#7f1d1d")
    ax_pqrst.set_xlabel("Time (s)")
    ax_pqrst.set_ylabel("Amplitude (mV)")
    ax_pqrst.set_title("A single, idealized heartbeat waveform")
    ax_pqrst.grid(alpha=0.3)
    st.pyplot(fig_pqrst, use_container_width=True)
    plt.close(fig_pqrst)

    col_a, col_b = st.columns(2)
    with col_a:
        with st.expander("🔗 Continuous-time model — x(t)", expanded=True):
            st.markdown(
                """
- **Notation:** $x : \\mathbb{R} \\to \\mathbb{R}$, i.e. $x(t)$ is defined for
  *every* real value of $t$ — no gaps.
- **Produced naturally by the body:** ion currents across cardiac cell
  membranes create a continuously varying voltage.
- **Infinite time resolution:** between any two instants, the signal has a
  value; it is never "undefined".
- **Analog by nature:** before any electronic processing, the ECG is a
  physical, continuous electrical phenomenon.
- **Cannot be stored directly:** a computer needs discrete numbers, so
  $x(t)$ must eventually be *sampled*.
"""
            )
        with st.expander("🏭 Acquisition chain"):
            st.markdown(
                """
**Heart → Electrodes → Amplifier & Filter → ADC → Digital ECG**

1. **Heart** generates the bioelectric signal.
2. **Electrodes** detect the voltage on the skin.
3. **Amplifier & analog filter** boosts the (tiny) signal and removes gross noise.
4. **ADC (Analog-to-Digital Converter)** samples *and* quantizes $x(t)$ — this
   is the bridge between the continuous world $x(t)$ and the discrete world
   $x[n]$.
5. **Digital ECG** is stored on the computer as $x[n]$.
"""
            )
    with col_b:
        with st.expander("🎯 Sampling theorem — x[n] = x(nT)", expanded=True):
            st.latex(r"x[n] = x(nT)")
            st.markdown(
                """
- $n$ — sample index (integer)
- $T$ — sampling period (seconds between samples)
- $f_s = 1/T$ — sampling frequency (Hz)

**Nyquist criterion:** $f_s$ must exceed **twice** the highest frequency
present in the ECG to avoid *aliasing*. Clinical ECGs are typically sampled
at **250–1000 Hz**, comfortably above the heart's ~0.5–40 Hz content.
"""
            )
        with st.expander("↔️ Continuous vs. discrete ECG"):
            st.markdown(
                """
| Continuous-time — x(t) | Discrete-time — x[n] |
|---|---|
| Defined for every real $t$ | Defined only at integer indices $n$ |
| Produced naturally by the heart | Produced by sampling $x(t)$ every $T$ seconds |
| Cannot be stored as-is | Stored, filtered, analyzed digitally |
| Infinite precision | Finite precision (quantization) |
| Domain: analog electronics | Domain: digital signal processing |
"""
            )

    st.subheader("Why digitizing the ECG matters")
    apps = [
        ("🧹 Noise & Artifact Filtering", "Digital filters remove baseline wander, muscle noise, and 50/60 Hz power-line interference."),
        ("📍 QRS Detection", "Algorithms (e.g. Pan–Tompkins) locate the R-peak to precisely time each heartbeat."),
        ("💓 Heart Rate & HRV", "Time between R-peaks gives instantaneous heart rate and variability analysis."),
        ("💾 Storage & Transmission", "Digital ECG can be compressed, stored, and sent remotely for telemedicine."),
        ("🧠 Arrhythmia Detection", "Pattern recognition / ML can classify abnormal rhythms automatically."),
        ("📡 Real-Time Monitoring", "Wearables and bedside monitors process x[n] continuously for instant alerts — including detecting asystole and triggering a code alarm."),
    ]
    cols = st.columns(3)
    for i, (title, desc) in enumerate(apps):
        with cols[i % 3]:
            st.markdown(f"**{title}**  \n{desc}")

# ---------------------------------------------------------------------
# TAB 2 — CHOOSE, LISTEN & ANALYZE  (the core requested flow)
# ---------------------------------------------------------------------
with tab_listen:
    st.markdown(
        "<span class='step-badge'>1</span>**Choose a heartbeat example**",
        unsafe_allow_html=True,
    )
    heartbeat_type = st.radio(
        "Pick one of the five rhythms",
        list(HEARTBEAT_TYPES.keys()),
        format_func=lambda k: f"{HEARTBEAT_TYPES[k]['icon']}  {k}",
        horizontal=False,
        label_visibility="collapsed",
        key="heartbeat_choice",
    )
    cfg = HEARTBEAT_TYPES[heartbeat_type]
    flat = is_flatline(heartbeat_type)

    st.markdown(f"### {cfg['icon']} {heartbeat_type}")
    st.write(cfg["description"])

    # One fixed rhythm (ground truth) per type — reused for the live
    # monitor, the audio, and the analysis, so everything you see, hear
    # and measure is the exact same heartbeat.
    N_BEATS = 40
    SEED = 7
    live_signal, rr_list = build_signal_for_kind(heartbeat_type, fs=150, n_beats=N_BEATS, seed=SEED)

    col1, col2, col3 = st.columns(3)
    if flat:
        col1.metric("Ground-truth Heart Rate", "0 BPM")
        col2.metric("Beats generated", "0")
        col3.metric("Status", "Asystole")
    else:
        true_hr = 60.0 / np.mean(rr_list)
        col1.metric("Ground-truth Heart Rate", f"{true_hr:.0f} BPM")
        col2.metric("Beats generated", f"{len(rr_list)}")
        col3.metric("Avg. RR interval", f"{np.mean(rr_list):.2f} s")

    st.divider()
    st.markdown(
        "<span class='step-badge'>2</span>**Listen & Watch**",
        unsafe_allow_html=True,
    )

    st.markdown("#### 📟 Live ECG Monitor")
    if flat:
        st.markdown(
            "<div class='flatline-alert'>🚨 CODE BLUE — the trace has gone flat. "
            "No P-QRS-T activity is present anywhere in the signal.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption(
            "Continuously sweeps like a real bedside monitor — the pen writes "
            "new beats and wraps around, overwriting the old trace, forever."
        )
    render_live_ecg_monitor(heartbeat_type, live_signal)

    st.markdown("#### 🔊 Listen to this heartbeat")
    if flat:
        st.caption(
            "This is the continuous **flatline alarm tone** a hospital monitor "
            "emits the instant it stops detecting a heartbeat — a single, "
            "unbroken pitch, very different from the rhythmic 'lub-dub' of a "
            "beating heart."
        )
    audio_bytes = synthesize_heartbeat_audio(heartbeat_type, rr_list)
    st.audio(audio_bytes, format="audio/wav")

    st.divider()
    st.markdown(
        "<span class='step-badge'>3</span>**Analyze this heartbeat**",
        unsafe_allow_html=True,
    )
    analyze_clicked = st.button("▶ Analyze This Heartbeat", key="analyze_btn")

    if analyze_clicked:
        fs_analysis = 500  # fine internal rate for accurate filtering/detection
        train, rr_list_analysis = build_signal_for_kind(
            heartbeat_type, fs=fs_analysis, n_beats=N_BEATS, seed=SEED
        )
        train_t = np.arange(len(train)) / fs_analysis

        filtered, peaks, rr, hr, sdnn, verdict, css_class = analyze_kind(
            heartbeat_type, train, fs_analysis
        )

        if flat:
            st.markdown(
                "<div class='flatline-alert'>🚨 <b>CODE BLUE — Asystole detected.</b> "
                "Sampling and filtering were run exactly as with the other rhythms, "
                "but R-peak detection correctly finds <b>zero</b> heartbeats — there is "
                "nothing periodic left to detect. In a real clinical device, this state "
                "triggers a continuous audible alarm and an emergency ('code blue') "
                "response.</div>",
                unsafe_allow_html=True,
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Detected R-peaks", "0")
            m2.metric("Detected Heart Rate", "0 BPM")
            m3.metric("Avg. RR interval", "—")
            m4.metric("RR variability (SDNN)", "—")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Detected R-peaks", f"{len(peaks)}")
            m2.metric("Detected Heart Rate", f"{hr:.1f} BPM" if hr else "—")
            m3.metric("Avg. RR interval", f"{np.mean(rr):.3f} s" if rr is not None else "—")
            m4.metric("RR variability (SDNN)", f"{sdnn:.1f} ms" if sdnn else "—")
            st.markdown(f"**Interpretation:** <span class='{css_class}'>{verdict}</span>", unsafe_allow_html=True)

        fig_a, ax_a = plt.subplots(figsize=(11, 3.2))
        plot_color = "#ef4444" if flat else cfg["color"]
        ax_a.plot(train_t, filtered, color=plot_color, linewidth=1.3, label="Filtered ECG")
        if len(peaks) > 0:
            ax_a.scatter(train_t[peaks], filtered[peaks], color="black", s=30, zorder=3, label="R Peaks")
        ax_a.set_xlim(0, min(10, train_t[-1]))
        ax_a.set_xlabel("Time (s)")
        ax_a.set_ylabel("Amplitude")
        ax_a.set_title(f"Analysis — {heartbeat_type} (first 10 s shown)")
        ax_a.legend()
        ax_a.grid(alpha=0.3)
        st.pyplot(fig_a, use_container_width=True)
        plt.close(fig_a)
    else:
        st.info("Press **Analyze This Heartbeat** to run sampling → filtering → R-peak detection → heart rate on this rhythm.")

    st.divider()
    st.markdown(
        "<span class='step-badge'>4</span>**Compare with the other rhythms** — "
        "head over to the **⚖️ Compare All 5 Types** tab to see this rhythm "
        "measured side-by-side against the other four, including the flatline case.",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------
# TAB 3 — SAMPLING & FILTERING (real recording)
# ---------------------------------------------------------------------
with tab_sampling:
    st.subheader("Working with the real, uploaded ECG recording")

    if not csv_ok:
        st.error("Could not load ecg_sample.csv")
        st.write(csv_error)
        st.stop()

    st.success(f"ECG dataset loaded successfully — {len(real_ecg)} samples.")
    real_time = np.arange(len(real_ecg)) / fs
    st.caption(f"Number of samples: {len(real_ecg)}  •  Duration at {fs} Hz: {len(real_ecg)/fs:.2f} s")

    st.markdown("##### 1. Original continuous-style ECG signal x(t)")
    fig1, ax1 = plt.subplots(figsize=(12, 3.2))
    ax1.plot(real_time, real_ecg, color="#b91c1c")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Original ECG signal")
    ax1.grid(alpha=0.3)
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    st.markdown("##### 2. Sampling — x[n] = x(nT)")
    st.latex(r"x[n] = x(nT)")
    new_fs = st.slider(
        "Choose a (re)sampling frequency to visualize",
        min_value=20,
        max_value=int(fs),
        value=min(150, int(fs)),
        step=10,
    )
    step = max(1, round(fs / new_fs))
    sampled_ecg = real_ecg[::step]
    sampled_time = real_time[::step]

    fig2, ax2 = plt.subplots(figsize=(12, 3.2))
    ax2.plot(real_time, real_ecg, label="Original ECG (dense)", alpha=0.6)
    ax2.scatter(sampled_time, sampled_ecg, s=16, color="#ef4444", label=f"Samples @ {new_fs} Hz", zorder=3)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Continuous signal vs. its discrete samples")
    ax2.legend()
    ax2.grid(alpha=0.3)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    if new_fs < 80:
        st.warning(
            "This sampling rate is well below the recommended clinical minimum "
            "(≥250 Hz) — notice how the waveform shape starts to blur. This is "
            "a hands-on look at the Nyquist criterion."
        )

    st.markdown("##### 3. Digital band-pass filtering (0.5–40 Hz)")
    filtered_real = bandpass_filter(real_ecg, fs)
    fig3, ax3 = plt.subplots(figsize=(12, 3.2))
    ax3.plot(real_time, filtered_real, color="#0f766e")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude")
    ax3.set_title("Filtered ECG signal")
    ax3.grid(alpha=0.3)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    st.session_state["_filtered_real"] = filtered_real

# ---------------------------------------------------------------------
# TAB 4 — R-PEAK DETECTION & HEART RATE (real recording)
# ---------------------------------------------------------------------
with tab_rpeak:
    st.subheader("R-peak detection & heart-rate estimation")

    if not csv_ok:
        st.error("Real ECG data is unavailable — see the Sampling & Filtering tab.")
        st.stop()

    filtered_real = st.session_state.get("_filtered_real", bandpass_filter(real_ecg, fs))
    real_time = np.arange(len(real_ecg)) / fs
    peaks = detect_r_peaks(filtered_real, fs, mode="real")

    fig4, ax4 = plt.subplots(figsize=(12, 3.4))
    ax4.plot(real_time, filtered_real, label="Filtered ECG", color="#0f766e")
    ax4.scatter(real_time[peaks], filtered_real[peaks], color="#ef4444", s=45, zorder=3, label="R Peaks")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Amplitude")
    ax4.set_title("Detected R Peaks")
    ax4.legend()
    ax4.grid(alpha=0.3)
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)

    rr, hr, sdnn = heart_rate_metrics(peaks, fs)
    c1, c2, c3 = st.columns(3)
    c1.metric("Detected R-peaks", f"{len(peaks)}")
    c2.metric("Estimated Heart Rate", f"{hr:.1f} BPM" if hr else "—")
    c3.metric("Avg. RR interval", f"{np.mean(rr):.3f} s" if rr is not None else "—")

    if hr is None:
        st.warning("Not enough R-peaks were detected to calculate heart rate.")
    else:
        verdict, css_class = verdict_for(hr, sdnn)
        st.markdown(f"**Interpretation:** <span class='{css_class}'>{verdict}</span>", unsafe_allow_html=True)

    st.divider()
    st.caption("Educational project — not intended for medical diagnosis.")

# ---------------------------------------------------------------------
# TAB 5 — COMPARE ALL 5 HEARTBEAT TYPES
# ---------------------------------------------------------------------
with tab_compare:
    st.subheader("How does each heartbeat type compare to a normal, healthy rhythm?")
    st.write(
        "Every type below is generated, filtered and analyzed with the exact "
        "same pipeline (sampling → band-pass filter → R-peak detection → "
        "heart rate), so the differences you see are purely due to rhythm, "
        "not the algorithm. The flatline case is included so you can see, "
        "numerically, exactly how 'no heartbeat' differs from every other "
        "rhythm — zero detected peaks, zero BPM."
    )

    fs_cmp = 500
    rows = []
    signals = {}
    for kind, kcfg in HEARTBEAT_TYPES.items():
        sig, rr_list_k = build_signal_for_kind(kind, fs=fs_cmp, n_beats=14, seed=7)
        filt, pk, rr, hr, sdnn, verdict, _ = analyze_kind(kind, sig, fs_cmp)
        signals[kind] = (sig, filt, pk)
        rows.append(
            {
                "Heartbeat Type": f"{kcfg['icon']} {kind}",
                "Detected BPM": round(hr, 1) if hr else 0,
                "Expected BPM range": (
                    "0 (no heartbeat)" if kcfg["flatline"]
                    else (f"{kcfg['expected_bpm'][0]}–{kcfg['expected_bpm'][1]}" if kcfg["expected_bpm"] else "highly variable")
                ),
                "RR variability (SDNN, ms)": round(sdnn, 1) if sdnn else None,
                "Verdict": verdict,
            }
        )

    df_compare = pd.DataFrame(rows)
    st.dataframe(df_compare, use_container_width=True, hide_index=True)

    st.markdown("##### Detected heart rate vs. the normal reference band (60–100 BPM)")
    fig_bar, ax_bar = plt.subplots(figsize=(9, 4))
    labels = list(HEARTBEAT_TYPES.keys())
    bpms = [row["Detected BPM"] or 0 for row in rows]
    colors = [HEARTBEAT_TYPES[k]["color"] for k in labels]
    ax_bar.axhspan(60, 100, color="#22c55e", alpha=0.15, label="Normal range (60–100 BPM)")
    bars = ax_bar.bar(labels, bpms, color=colors)
    for bar, kind in zip(bars, labels):
        if HEARTBEAT_TYPES[kind]["flatline"]:
            ax_bar.annotate("ASYSTOLE", (bar.get_x() + bar.get_width() / 2, 2),
                             ha="center", fontsize=9, fontweight="bold", color="#ef4444")
    ax_bar.set_ylabel("Heart Rate (BPM)")
    ax_bar.set_xticks(range(len(labels)))
    ax_bar.set_xticklabels(labels, rotation=15, ha="right")
    ax_bar.legend()
    ax_bar.grid(alpha=0.3, axis="y")
    st.pyplot(fig_bar, use_container_width=True)
    plt.close(fig_bar)

    st.markdown("##### Waveform comparison")
    fig_multi, axes = plt.subplots(len(HEARTBEAT_TYPES), 1, figsize=(12, 11), sharex=False)
    for ax, (kind, kcfg) in zip(axes, HEARTBEAT_TYPES.items()):
        sig, filt, pk = signals[kind]
        t_axis = np.arange(len(filt)) / fs_cmp
        line_color = "#ef4444" if kcfg["flatline"] else kcfg["color"]
        ax.plot(t_axis, filt, color=line_color, linewidth=1.2)
        if len(pk) > 0:
            ax.scatter(t_axis[pk], filt[pk], color="black", s=18, zorder=3)
        ax.set_title(kind, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 8)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    fig_multi.tight_layout()
    st.pyplot(fig_multi, use_container_width=True)
    plt.close(fig_multi)

    st.info(
        "💡 **How to read this:** the shaded green band marks a healthy resting "
        "heart rate. Tachycardia sits above it, bradycardia sits below it, the "
        "irregular rhythm may sit inside the band on average while still "
        "showing a much larger RR variability (SDNN) — which is what actually "
        "makes it 'irregular' — and the flatline case sits at exactly 0 BPM "
        "with no RR variability to measure at all, because there are no beats."
    )

st.divider()
st.caption(
    "ECG as a Signal — Signals & Systems Project • Built with Streamlit, "
    "NumPy, SciPy and Matplotlib • Educational use only, not medical advice."
)
