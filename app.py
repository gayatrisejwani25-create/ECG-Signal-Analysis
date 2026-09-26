import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import io
from scipy.io.wavfile import write

st.set_page_config(
    page_title="ECG Signal Analysis",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ ECG Signal Analysis")
st.subheader("Signals and Systems Project")

st.write(
    "This project demonstrates ECG signal visualization, "
    "sampling, digital filtering, R-peak detection and "
    "heart-rate estimation."
)

st.divider()

# ==========================================
# LOAD ECG DATA
# ==========================================

st.header("1. ECG Signal")

try:

    data = pd.read_csv("ecg_sample.csv",on_bad_lines="skip",engine="python")

    st.success("ECG dataset loaded successfully!")

except Exception as e:

    st.error("Could not load ecg_sample.csv")

    st.write(e)

    st.stop()


# ==========================================
# SHOW DATA
# ==========================================

st.write("First five rows of ECG data:")

st.dataframe(data.head())

# Play heartbeat sound
st.subheader("🔊 Listen to Heartbeat")

if st.button("▶ Play Heartbeat"):
    sample_rate = 44100
    duration = 30.0

    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        endpoint=False
    )

    audio = np.zeros_like(t)

    def add_beat(start, strength=1.0):
        idx = t >= start
        local_t = t[idx] - start

        envelope = np.exp(-35 * local_t)

        lub = np.sin(2 * np.pi * 70 * local_t)
        dub = np.sin(2 * np.pi * 110 * local_t)

        audio[idx] += strength * envelope * (
            0.8 * lub + 0.4 * dub
        )

   for start in np.arange(0, duration, 0.83):
    add_beat(start, 1.0)

    if start + 0.28 < duration:
        add_beat(start + 0.28, 0.7)

    audio = audio / np.max(np.abs(audio))
    audio = (audio * 32767).astype(np.int16)

    buffer = io.BytesIO()
    write(buffer, sample_rate, audio)

    st.audio(buffer.getvalue(), format="audio/wav")

# ==========================================
# SELECT ECG COLUMN
# ==========================================

ecg_column = data.columns[0]

data[ecg_column] = pd.to_numeric(
    data[ecg_column],
    errors="coerce"
)

data = data.dropna(subset=[ecg_column])

signal = data[ecg_column].to_numpy()


ecg=signal


# ==========================================
# SAMPLING FREQUENCY
# ==========================================

fs = st.number_input(
    "Sampling frequency (Hz)",
    min_value=50,
    max_value=2000,
    value=360,
    step=10
)


# ==========================================
# TIME AXIS
# ==========================================

time = np.arange(
    len(ecg)
) / fs


st.write(
    f"Number of samples: {len(ecg)}"
)

st.write(
    f"Signal duration: {len(ecg) / fs:.2f} seconds"
)


# ==========================================
# ORIGINAL ECG
# ==========================================

st.header("2. Original ECG Signal")

fig, ax = plt.subplots(
    figsize=(12, 4)
)

ax.plot(
    time,
    ecg
)

ax.set_xlabel(
    "Time (seconds)"
)

ax.set_ylabel(
    "Amplitude"
)

ax.set_title(
    "Original ECG Signal x(t)"
)

ax.grid(True)

st.pyplot(fig)


# ==========================================
# SAMPLING
# ==========================================

st.header("3. Sampling")

st.latex(
    r"x[n] = x(nT)"
)

st.write(
    "The continuous ECG signal is represented "
    "using discrete samples."
)

new_fs = st.slider(
    "Choose sampling frequency",
    min_value=50,
    max_value=int(fs),
    value=min(250, int(fs)),
    step=10
)

step = max(
    1,
    round(fs / new_fs)
)

sampled_ecg = ecg[::step]

sampled_time = time[::step]


fig2, ax2 = plt.subplots(
    figsize=(12, 4)
)

ax2.plot(
    time,
    ecg,
    label="Original ECG"
)

ax2.scatter(
    sampled_time,
    sampled_ecg,
    s=15,
    label="Samples"
)

ax2.set_xlabel(
    "Time (seconds)"
)

ax2.set_ylabel(
    "Amplitude"
)

ax2.set_title(
    "ECG Sampling"
)

ax2.legend()

ax2.grid(True)

st.pyplot(fig2)


# ==========================================
# DIGITAL FILTER
# ==========================================

st.header("4. Digital Filtering")

nyquist = fs / 2

low_cutoff = 0.5
high_cutoff = 40

if high_cutoff < nyquist:

    low = low_cutoff / nyquist
    high = high_cutoff / nyquist

    b, a = butter(
        4,
        [low, high],
        btype="bandpass"
    )

    filtered_ecg = filtfilt(
        b,
        a,
        ecg
    )

else:

    filtered_ecg = ecg


fig3, ax3 = plt.subplots(
    figsize=(12, 4)
)

ax3.plot(
    time,
    filtered_ecg
)

ax3.set_xlabel(
    "Time (seconds)"
)

ax3.set_ylabel(
    "Amplitude"
)

ax3.set_title(
    "Filtered ECG Signal"
)

ax3.grid(True)

st.pyplot(fig3)


# ==========================================
# R PEAK DETECTION
# ==========================================

st.header("5. R-Peak Detection")

distance = int(
    0.4 * fs
)

prominence = (
    0.5 * np.std(filtered_ecg)
)

peaks, properties = find_peaks(
    filtered_ecg,
    distance=distance,
    prominence=prominence
)


fig4, ax4 = plt.subplots(
    figsize=(12, 4)
)

ax4.plot(
    time,
    filtered_ecg,
    label="Filtered ECG"
)

ax4.scatter(
    time[peaks],
    filtered_ecg[peaks],
    label="R Peaks"
)

ax4.set_xlabel(
    "Time (seconds)"
)

ax4.set_ylabel(
    "Amplitude"
)

ax4.set_title(
    "Detected R Peaks"
)

ax4.legend()

ax4.grid(True)
st.write(f"Number of detected R peaks: {len(peaks)}")

if len(peaks) > 1:
    rr_intervals = np.diff(peaks) / fs
    heart_rate = 60 / np.mean(rr_intervals)
    st.metric("Heart Rate", f"{heart_rate:.1f} BPM")
else:
    st.warning("Not enough R-peaks detected to calculate heart rate.")

st.pyplot(fig4)





# ==========================================
# HEART RATE
# ==========================================

st.header("6. Heart Rate")

if len(peaks) >= 2:

    rr_intervals = (
        np.diff(peaks) / fs
    )

    average_rr = np.mean(
        rr_intervals
    )

    heart_rate = (
        60 / average_rr
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Estimated Heart Rate",
            f"{heart_rate:.2f} BPM"
        )

    with col2:

        st.metric(
            "Average RR Interval",
            f"{average_rr:.3f} seconds"
        )

else:

    st.warning(
        "Not enough R-peaks were detected."
    )


st.divider()

st.success(
    "ECG signal analysis completed."
)

st.caption(
    "Educational project — not intended for medical diagnosis."
)
