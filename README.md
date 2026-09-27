# ❤️ ECG as a Signal (v2 — Reference vs. Sample edition)

### Signals & Systems Project — interactive Streamlit app

This project models the complete path of the heart's electrical activity
from its analog origin to a computer-ready digital representation, and lets
you **choose**, **listen to**, **watch**, **analyze**, and **compare** five
different heartbeat rhythms — including what happens when the heart stops —
against a fixed, healthy **reference rhythm**.

```
Heart → ECG Sensor → Continuous Signal x(t) → Sampling → Digital ECG x[n]
      → Filtering → R-Peak Detection → Heart Rate → Reference Comparison
```

## ✨ What's new in v2

- **Reference vs. Sample audio.** Every time you pick a heartbeat sample,
  you also hear the fixed **Normal Heartbeat reference** audio right next
  to it, each with its own **download button** (`.wav`).
- **Real, working reference links.** Clickable links to genuine,
  human-recorded ECG/heart-sound material appear next to the audio and in a
  "References & Further Reading" section at the end of the app:
  - [PhysioNet — MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
  - [Wikipedia — Electrocardiography](https://en.wikipedia.org/wiki/Electrocardiography)
  - [Wikipedia — Heart sounds](https://en.wikipedia.org/wiki/Heart_sounds)
- **Four new comparison graphs**, generated after pressing "Analyze This
  Heartbeat", all using distinct, consistent colors (🔵 blue = reference,
  the rhythm's own color = sample):
  1. **Continuous signal comparison** — reference x(t) vs. sample x(t).
  2. **Discrete signal comparison** — reference x[n] vs. sample x[n], with
     an adjustable sampling-frequency slider.
  3. **Combined dashboard** — the continuous and discrete comparisons
     stacked together for a single at-a-glance view.
  4. (Existing) filtered signal + detected R-peaks for the chosen sample.
- **Quantitative similarity metrics** — a normalized cross-correlation
  "closeness score" (ρ) computed on both the continuous and discrete
  signals, plus an **R-peak amplitude-retention** percentage that exposes
  real undersampling: drop the sampling-frequency slider low enough and
  you'll see the sharp R-wave peak stop being captured accurately, a
  hands-on demonstration of the **Nyquist criterion**.
- **New theory section** ("🆚 Comparing a Reference Signal to a Sample
  Signal") explaining the math behind the similarity score.
- **Performance:** heartbeat-signal generation and audio synthesis are now
  `@st.cache_data`-cached, so re-selecting a rhythm or dragging a slider no
  longer recomputes everything from scratch on every rerun.
- **Bug fix:** the "Analyze" section (including the new comparison graphs)
  now stays visible while you adjust the new sliders, instead of vanishing
  on the next rerun (a common Streamlit `st.button` pitfall, fixed with
  `st.session_state`).

## ✨ Original features (still included)

- **Theory walkthrough** — what an ECG is, the continuous-time model
  `x(t)`, the acquisition chain, the sampling theorem `x[n] = x(nT)`, the
  Nyquist criterion, and why digitizing the ECG matters.
- **A guided flow** in the "Choose, Listen & Analyze" tab: Choose → Listen
  & Watch (now: reference + sample) → Analyze (now: + reference
  comparison) → Compare with all five types.
- **5 audible + visible heartbeat examples**: Normal, Tachycardia,
  Bradycardia, Irregular, and Cardiac Arrest (Flatline/Asystole).
- **Sampling & filtering demo on a real ECG recording** (`ecg_sample.csv`).
- **R-peak detection & heart-rate estimation**, with RR-interval and SDNN
  (variability) metrics.
- **Side-by-side comparison of all five heartbeat types** against the
  normal 60–100 BPM reference band.

## 🗂️ Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit application (all logic lives here) |
| `ecg_sample.csv` | A real single-column ECG recording used for the sampling/filtering/R-peak demo |
| `requirements.txt` | Python dependencies (unchanged — no new packages needed) |
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

## ⚠️ Disclaimer

This project is for **educational purposes** as part of a Signals &
Systems course and is **not intended for medical diagnosis**.
