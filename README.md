# ❤️ ECG as a Signal

### Signals & Systems Project — interactive Streamlit app

This project models the complete path of the heart's electrical activity
from its analog origin to a computer-ready digital representation, and lets
you **listen to**, **watch**, and **compare** four different heartbeat
rhythms.

```
Heart → ECG Sensor → Continuous Signal x(t) → Sampling → Digital ECG x[n]
      → Filtering → R-Peak Detection → Heart Rate
```

## ✨ Features

- **Theory walkthrough** — what an ECG is, the continuous-time model
  `x(t)`, the acquisition chain, the sampling theorem `x[n] = x(nT)`, the
  Nyquist criterion, and why digitizing the ECG matters.
- **4 audible + visible heartbeat examples**
  - 💚 Normal Heartbeat (60–100 BPM)
  - 🔴 Tachycardia — High Rate (100–160 BPM)
  - 🔵 Bradycardia — Low Rate (30–55 BPM)
  - 🟠 Irregular / arrhythmia-like rhythm (highly variable RR intervals)

  Each example is synthesized from a realistic P‑Q‑R‑S‑T waveform model,
  can be **played back as audio**, and shown as a **moving "ECG monitor"
  style live trace**.
- **Sampling & filtering demo on a real ECG recording** (`ecg_sample.csv`)
  — adjustable sampling frequency, and a 4th-order Butterworth band-pass
  filter (0.5–40 Hz).
- **R-peak detection & heart-rate estimation** on the real recording,
  with RR-interval and SDNN (variability) metrics.
- **Side-by-side comparison** of all four heartbeat types against the
  normal 60–100 BPM reference band, with a comparison table and stacked
  waveform plots — so you can see exactly how each condition deviates
  from "how a heartbeat should look".

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

## 🧠 How the four heartbeat examples are generated

Each heartbeat type is built from a standard **5-Gaussian ECG beat model**
(P, Q, R, S, T waves), repeated back-to-back at different RR intervals:

| Type | Typical RR interval | Resulting heart rate |
|---|---|---|
| Normal | ~0.80–1.00 s, low jitter | 60–100 BPM |
| Tachycardia | ~0.40–0.50 s | 100–160 BPM |
| Bradycardia | ~1.35–1.55 s | 30–55 BPM |
| Irregular | 0.45–1.55 s, randomized every beat | highly variable |

The **same** sampling → band-pass filter → R-peak detection pipeline is
then run on every type, so the app both *demonstrates* and *validates* the
signal-processing chain taught in the course.

The real `ecg_sample.csv` recording is used separately to show the same
pipeline working on genuine (noisier, less "textbook") ECG data.

## ⚠️ Disclaimer

This project is for **educational purposes** as part of a Signals &
Systems course and is **not intended for medical diagnosis**.
