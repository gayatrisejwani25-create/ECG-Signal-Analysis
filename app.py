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
2. Four audible + visible heartbeat examples: Normal, Tachycardia
   (fast), Bradycardia (slow) and an Irregular / arrhythmia-like rhythm,
   plus a moving "ECG monitor" style live trace.
3. A real ECG recording (ecg_sample.csv) used to demonstrate sampling,
   digital band-pass filtering and R-peak / heart-rate detection.
4. A side-by-side comparison of all four heartbeat types against the
   normal reference range, so you can see how each one deviates from
   "how a heartbeat should look".

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
    .metric-good {color: #22c55e; font-weight: 700;}
    .metric-warn {color: #f59e0b; font-weight: 700;}
    .metric-bad {color: #ef4444; font-weight: 700;}
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

HEARTBEAT_TYPES = {
    "Normal Heartbeat": {
        "icon": "💚",
        "color": "#22c55e",
        "base_rr": 0.90,
        "jitter": 0.015,
        "range": (0.80, 1.00),
        "expected_bpm": (60, 100),
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
        "description": (
            "An irregular rhythm — RR intervals swing unpredictably, "
            "similar in spirit to an arrhythmia such as AFib."
        ),
    },
}

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
    if len(peaks) < 2:
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
# Synthetic multi-beat ECG generator (used for the 4 heartbeat examples)
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
    """Generate a ground-truth list of RR intervals (seconds) for a heartbeat type."""
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


def render_live_ecg_monitor(kind: str, rr_list: list, columns_per_second: int = 150, height: int = 260):
    """
    Render a genuinely MOVING, continuously-sweeping ECG monitor — like a
    real bedside cardiac monitor — using an HTML5 <canvas> animated with
    requestAnimationFrame (runs entirely in the browser, so it never
    freezes or "finishes" the way a Python-side loop does).

    The trace is drawn as a moving pen: new samples are written just ahead
    of a small blank gap, and once the sweep reaches the right edge it
    wraps back to the left and starts overwriting the old trace — exactly
    like a real ECG / vitals monitor.
    """
    cfg = HEARTBEAT_TYPES[kind]

    # Build the waveform at a sample rate that matches the animation speed
    # (1 real second of playback == 1 second of signal), so faster rhythms
    # (tachycardia) visibly sweep by quicker than slower ones (bradycardia).
    stream = build_ecg_train(rr_list, columns_per_second)
    amp = np.max(np.abs(stream)) or 1.0
    normalized = (stream / amp).tolist()
    data_json = json.dumps(normalized)

    canvas_id = f"ecgCanvas_{abs(hash(kind)) % 100000}"

    template = """
    <div style="background:#020617;border-radius:14px;padding:10px 14px;
                border:1px solid #1f2937;">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:6px;">
        <span style="color:#9ca3af;font:600 12px sans-serif;letter-spacing:.6px;">
          &#128137; LIVE ECG MONITOR
        </span>
        <span style="color:__COLOR__;font:700 13px sans-serif;">__LABEL__</span>
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
        const gapAhead = 14;            // blank "pen tip" gap ahead of the trace

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
        .replace("__COLOR__", cfg["color"])
        .replace("__LABEL__", kind.upper())
        .replace("__CANVAS_ID__", canvas_id)
        .replace("__CANVAS_H__", str(height))
    )
    components_html(html_code, height=height + 60)


def synthesize_heartbeat_audio(rr_intervals: list, sample_rate: int = AUDIO_FS):
    """Turn a list of RR intervals into an audible 'lub-dub' heartbeat sound."""
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

    heartbeat_type = st.selectbox(
        "Heartbeat example",
        list(HEARTBEAT_TYPES.keys()),
        help="Used in the 'Listen & Watch' and comparisons.",
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
        "🔊 Listen & Watch",
        "🧮 Sampling & Filtering",
        "📈 R-Peaks & Heart Rate",
        "⚖️ Compare All 4 Types",
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
        ("📡 Real-Time Monitoring", "Wearables and bedside monitors process x[n] continuously for instant alerts."),
    ]
    cols = st.columns(3)
    for i, (title, desc) in enumerate(apps):
        with cols[i % 3]:
            st.markdown(f"**{title}**  \n{desc}")

# ---------------------------------------------------------------------
# TAB 2 — LISTEN & WATCH
# ---------------------------------------------------------------------
with tab_listen:
    cfg = HEARTBEAT_TYPES[heartbeat_type]
    st.subheader(f"{cfg['icon']} {heartbeat_type}")
    st.write(cfg["description"])

    # One fixed rhythm (ground truth) per type — reused for the live
    # monitor, the audio, and the analysis, so everything you see, hear
    # and measure is the exact same heartbeat.
    N_BEATS = 40
    SEED = 7
    rr_list = rr_sequence(heartbeat_type, n_beats=N_BEATS, seed=SEED)
    true_hr = 60.0 / np.mean(rr_list)

    col1, col2, col3 = st.columns(3)
    col1.metric("Ground-truth Heart Rate", f"{true_hr:.0f} BPM")
    col2.metric("Beats generated", f"{len(rr_list)}")
    col3.metric("Avg. RR interval", f"{np.mean(rr_list):.2f} s")

    st.markdown("#### 📟 Live ECG Monitor")
    st.caption(
        "Continuously sweeps like a real bedside monitor — the pen writes "
        "new beats and wraps around, overwriting the old trace, forever."
    )
    render_live_ecg_monitor(heartbeat_type, rr_list)

    st.markdown("#### 🔊 Step 1 — Listen to this heartbeat")
    audio_bytes = synthesize_heartbeat_audio(rr_list)
    st.audio(audio_bytes, format="audio/wav")

    st.markdown("#### 🔬 Step 2 — Analyze this heartbeat")
    analyze_clicked = st.button("▶ Analyze This Heartbeat", key="analyze_btn")

    if analyze_clicked:
        fs_analysis = 500  # fine internal rate for accurate filtering/detection
        train = build_ecg_train(rr_list, fs_analysis)
        train_t = np.arange(len(train)) / fs_analysis

        filtered = bandpass_filter(train, fs_analysis)
        peaks = detect_r_peaks(filtered, fs_analysis, mode="synth")
        rr, hr, sdnn = heart_rate_metrics(peaks, fs_analysis)
        verdict, css_class = verdict_for(hr, sdnn)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Detected R-peaks", f"{len(peaks)}")
        m2.metric("Detected Heart Rate", f"{hr:.1f} BPM" if hr else "—")
        m3.metric("Avg. RR interval", f"{np.mean(rr):.3f} s" if rr is not None else "—")
        m4.metric("RR variability (SDNN)", f"{sdnn:.1f} ms" if sdnn else "—")

        st.markdown(f"**Interpretation:** <span class='{css_class}'>{verdict}</span>", unsafe_allow_html=True)

        fig_a, ax_a = plt.subplots(figsize=(11, 3.2))
        ax_a.plot(train_t, filtered, color=cfg["color"], linewidth=1.3, label="Filtered ECG")
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
# TAB 5 — COMPARE ALL 4 HEARTBEAT TYPES
# ---------------------------------------------------------------------
with tab_compare:
    st.subheader("How does each heartbeat type compare to a normal, healthy rhythm?")
    st.write(
        "Every type below is generated, filtered and analyzed with the exact "
        "same pipeline (sampling → band-pass filter → R-peak detection → "
        "heart rate), so the differences you see are purely due to rhythm, "
        "not the algorithm."
    )

    fs_cmp = 500
    rows = []
    signals = {}
    for kind, cfg in HEARTBEAT_TYPES.items():
        rr_list = rr_sequence(kind, n_beats=14)
        sig = build_ecg_train(rr_list, fs_cmp)
        filt = bandpass_filter(sig, fs_cmp)
        pk = detect_r_peaks(filt, fs_cmp, mode="synth")
        rr, hr, sdnn = heart_rate_metrics(pk, fs_cmp)
        verdict, _ = verdict_for(hr, sdnn)
        signals[kind] = (sig, filt, pk)
        rows.append(
            {
                "Heartbeat Type": f"{cfg['icon']} {kind}",
                "Detected BPM": round(hr, 1) if hr else None,
                "Expected BPM range": f"{cfg['expected_bpm'][0]}–{cfg['expected_bpm'][1]}" if cfg["expected_bpm"] else "highly variable",
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
    ax_bar.bar(labels, bpms, color=colors)
    ax_bar.set_ylabel("Heart Rate (BPM)")
    ax_bar.set_xticks(range(len(labels)))
    ax_bar.set_xticklabels(labels, rotation=15, ha="right")
    ax_bar.legend()
    ax_bar.grid(alpha=0.3, axis="y")
    st.pyplot(fig_bar, use_container_width=True)
    plt.close(fig_bar)

    st.markdown("##### Waveform comparison")
    fig_multi, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=False)
    for ax, (kind, cfg) in zip(axes, HEARTBEAT_TYPES.items()):
        sig, filt, pk = signals[kind]
        t_axis = np.arange(len(filt)) / fs_cmp
        ax.plot(t_axis, filt, color=cfg["color"], linewidth=1.2)
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
        "heart rate. Tachycardia sits above it, bradycardia sits below it, and "
        "the irregular rhythm may sit inside the band on average while still "
        "showing a much larger RR variability (SDNN) — which is what actually "
        "makes it 'irregular'."
    )

st.divider()
st.caption(
    "ECG as a Signal — Signals & Systems Project • Built with Streamlit, "
    "NumPy, SciPy and Matplotlib • Educational use only, not medical advice."
)
