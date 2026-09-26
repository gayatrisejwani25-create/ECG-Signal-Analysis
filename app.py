import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

st.set_page_config(
    page_title="ECG Signal Analysis",
    page_icon="❤️",
    layout="wide"
)

st.title("ECG as a Signal")
st.subheader("Modeling Cardiac Electrical Activity in Signals & Systems")

st.write(
    "This project demonstrates ECG signal visualization, "
    "sampling, filtering, R-peak detection and heart-rate estimation."
)

st.divider()

# ==============================
# UPLOAD ECG
# ==============================

st.header("1. Upload ECG Signal")

uploaded_file = st.file_uploader(
    "Upload an ECG CSV file",
    type=["csv"]
)

if uploaded_file is None:

    st.info(
        "Please upload an ECG CSV file to start the analysis."
    )

else:

    data = pd.read_csv(uploaded_file)

    st.success("ECG file uploaded successfully!")

    st.write("Dataset preview:")
    st.dataframe(data.head())

    # ==============================
    # SELECT SIGNAL
    # ==============================

    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns.tolist()

    if len(numeric_columns) == 0:

        st.error(
            "No numeric ECG signal column was found."
        )

    else:

        signal_column = st.selectbox(
            "Select ECG signal column",
            numeric_columns
        )

        ecg = data[signal_column].dropna().values

        # ==============================
        # SAMPLING FREQUENCY
        # ==============================

        fs = st.number_input(
            "Original Sampling Frequency (Hz)",
            min_value=50,
            max_value=2000,
            value=360,
            step=10
        )

        time = np.arange(len(ecg)) / fs

        st.write(
            f"Number of samples: {len(ecg)}"
        )

        st.write(
            f"Duration: {len(ecg) / fs:.2f} seconds"
        )

        # ==============================
        # ORIGINAL ECG
        # ==============================

        st.header("2. Original ECG Signal x(t)")

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(time, ecg)

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Original ECG Signal")

        ax.grid(True)

        st.pyplot(fig)

        # ==============================
        # SAMPLING
        # ==============================

        st.header("3. Sampling")

        st.latex(r"x[n] = x(nT)")

        st.write(
            "Choose a sampling frequency to observe "
            "the discrete-time representation."
        )

        new_fs = st.slider(
            "Sampling Frequency",
            min_value=50,
            max_value=int(fs),
            value=min(250, int(fs)),
            step=10
        )

        sample_step = max(
            1,
            round(fs / new_fs)
        )

        sampled_ecg = ecg[::sample_step]
        sampled_time = time[::sample_step]

        fig2, ax2 = plt.subplots(figsize=(12, 4))

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

        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Amplitude")

        ax2.set_title(
            "Continuous ECG and Sampled ECG"
        )

        ax2.legend()
        ax2.grid(True)

        st.pyplot(fig2)

        st.success(
            f"Sampling frequency = {new_fs} Hz"
        )

        # ==============================
        # FILTERING
        # ==============================

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

        else:

            filtered_ecg = ecg

            st.warning(
                "Sampling frequency is too low "
                "for the selected filter."
            )

        # ==============================
        # R PEAK DETECTION
        # ==============================

        st.header("5. R-Peak Detection")

        peak_distance = int(
            0.4 * fs
        )

        prominence = (
            0.5 * np.std(filtered_ecg)
        )

        peaks, properties = find_peaks(
            filtered_ecg,
            distance=peak_distance,
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
            color="red",
            label="R Peaks"
        )

        ax4.set_xlabel(
            "Time (seconds)"
        )

        ax4.set_ylabel(
            "Amplitude"
        )

        ax4.set_title(
            "R-Peak Detection"
        )

        ax4.legend()
        ax4.grid(True)

        st.pyplot(fig4)

        st.write(
            f"Detected R-peaks: {len(peaks)}"
        )

        # ==============================
        # HEART RATE
        # ==============================

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
                    f"{average_rr:.3f} s"
                )

        else:

            st.warning(
                "Not enough R-peaks detected "
                "to calculate heart rate."
            )

st.divider()

st.caption(
    "Educational Signals & Systems project. "
    "Not intended for medical diagnosis."
)
