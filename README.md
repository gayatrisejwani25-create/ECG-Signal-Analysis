# ❤️ ECG as a Signal

### Signals & Systems Project — interactive Streamlit app

This project models the complete path of the heart's electrical activity
from its analog origin to a computer-ready digital representation, and lets
you **choose**, **listen to**, **watch**, **analyze**, and **compare** five
different heartbeat rhythms — including what happens when the heart stops.

```
Heart → ECG Sensor → Continuous Signal x(t) → Sampling → Digital ECG x[n]
      → Filtering → R-Peak Detection → Heart Rate
```

## ✨ Features

- **Theory walkthrough** — what an ECG is, the continuous-time model
  `x(t)`, the acquisition chain, the sampling theorem `x[n] = x(nT)`, the
  Nyquist criterion, and why digitizing the ECG matters.
- **A guided 4-step flow** in the "Choose, Listen & Analyze" tab:
  1. **Choose** one of five heartbeat examples.
  2. **Listen & Watch** — hear it as audio and watch it sweep across a
     live, canvas-animated bedside-monitor-style trace.
  3. **Analyze** — run sampling → band-pass filtering → R-peak detection →
     heart-rate estimation on that exact rhythm.
  4. **Compare** — jump to the comparison tab to see it measured
     side-by-side against the other four.
- **5 audible + visible heartbeat examples**
  - 💚 Normal Heartbeat (60–100 BPM)
  - 🔴 Tachycardia — High Rate (100–160 BPM)
  - 🔵 Bradycardia — Low Rate (30–55 BPM)
  - 🟠 Irregular / arrhythmia-like rhythm (highly variable RR intervals)
  - ⚫ **Cardiac Arrest (Flatline / Asystole)** — a flat trace with no
    P‑QRS‑T activity, paired with a continuous monitor **alarm tone**
    instead of "lub-dub" beats, modeling what a real bedside monitor does
    the instant it loses a heartbeat. Analysis correctly reports **0
    detected R-peaks / 0 BPM** rather than forcing a false reading.

  The four living rhythms are synthesized from a realistic P‑Q‑R‑S‑T
  waveform model; the flatline example is a near-zero noise trace of the
  same duration. Every example can be **played back as audio** and shown
  as a **moving "ECG monitor" style live trace**.
- **Sampling & filtering demo on a real ECG recording** (`ecg_sample.csv`)
  — adjustable sampling frequency, and a 4th-order Butterworth band-pass
  filter (0.5–40 Hz).
- **R-peak detection & heart-rate estimation** on the real recording,
  with RR-interval and SDNN (variability) metrics.
- **Side-by-side comparison of all five heartbeat types** against the
  normal 60–100 BPM reference band, with a comparison table and stacked
  waveform plots — so you can see exactly how each condition (including
  asystole) deviates from "how a heartbeat should look".

## 🗂️ Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit application (all logic lives here) |
| `ecg_sample.csv` | A real single-column ECG recording used for the sampling/filtering/R-peak demo |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## 🚀 Running locally

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

## ☁️ Deploying on Streamlit Community Cloud

1. Push this repo to GitHub (make sure `app.py`, `ecg_sample.csv` and
   `requirements.txt` are all in the same folder).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and
   click **"New app"**.
3. Point it at your repo, branch, and `app.py` as the entry file.
4. Deploy — Streamlit Cloud installs `requirements.txt` automatically.

## 🧠 How the five heartbeat examples are generated

Four types are built from a standard **5-Gaussian ECG beat model** (P, Q,
R, S, T waves), repeated back-to-back at different RR intervals. The fifth
(flatline) is a near-zero noise trace with no beat model applied at all —
because asystole has no P‑QRS‑T activity to synthesize:

| Type | Typical RR interval | Resulting heart rate |
|---|---|---|
| Normal | ~0.80–1.00 s, low jitter | 60–100 BPM |
| Tachycardia | ~0.40–0.50 s | 100–160 BPM |
| Bradycardia | ~1.35–1.55 s | 30–55 BPM |
| Irregular | 0.45–1.55 s, randomized every beat | highly variable |
| Cardiac Arrest (Flatline) | no beats — flat trace | 0 BPM (asystole) |

The **same** sampling → band-pass filter → R-peak detection pipeline is
then run on every type. For the four living rhythms this both
*demonstrates* and *validates* the signal-processing chain taught in the
course; for the flatline case it demonstrates that the pipeline correctly
finds **zero** peaks rather than hallucinating a heartbeat from noise.

Audio is generated the same way for the four living rhythms (a "lub-dub"
S1/S2 heart-sound model timed to each RR interval), while the flatline
type instead plays a single continuous alarm tone — the same behavior a
real cardiac monitor exhibits when it detects asystole.

The real `ecg_sample.csv` recording is used separately to show the same
pipeline working on genuine (noisier, less "textbook") ECG data.

## ⚠️ Disclaimer

This project is for **educational purposes** as part of a Signals &
Systems course and is **not intended for medical diagnosis**. The
"Cardiac Arrest (Flatline)" example is a simplified educational model of
monitor behavior, not a medical simulation.
