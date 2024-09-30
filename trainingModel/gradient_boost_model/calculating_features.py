from __future__ import division
import pandas as pd
import numpy as np
import scipy.signal
from scipy.signal import welch
from scipy.stats import entropy
from numpy.fft import fft


def calculate_heart_rate_from_rpeaks(r_peaks, sampling_rate):
    # Convert indices to times
    r_peak_times = r_peaks / sampling_rate

    # Time between successive R-peaks
    heart_rate_intervals = pd.Series(r_peak_times).diff()

    # Convert intervals to BPM
    heart_rate_bpm = 60 / heart_rate_intervals.mean()

    return heart_rate_bpm

def calculate_morphological_features(templates_df):
    # Peak-to-peak amplitude
    peak_to_peak = templates_df.max(axis=1) - templates_df.min(axis=1)

    # Area under the curve
    auc = templates_df.apply(lambda row: np.trapz(row), axis=1)

    return pd.DataFrame({
        'peak_to_peak': peak_to_peak,
        'auc': auc
    })

def calculate_hrv_features(rr_intervals_ms):
    sdnn = np.std(rr_intervals_ms)

    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals_ms))))

    pnn50 = 100 * np.sum(np.abs(np.diff(rr_intervals_ms)) > 50) / len(rr_intervals_ms)

    sampling_frequency = 1 / np.mean(rr_intervals_ms) * 1000

    f, Pxx = scipy.signal.periodogram(rr_intervals_ms, fs=sampling_frequency, scaling='density')

    hf_band = (0.15, 0.4)

    hf_power = np.trapz([Pxx[(f >= hf_band[0]) & (f <= hf_band[1])]], f[(f >= hf_band[0]) & (f <= hf_band[1])])[0]

    return {
        'SDNN': sdnn,
        'RMSSD': rmssd,
        'pNN50': pnn50,
        'HF_power': hf_power
    }


def calculate_psd(signal, fs):
    nperseg = min(len(signal), 1024)

    pxx = welch(signal, fs=fs, nperseg=nperseg)

    psd_df = pd.DataFrame({
        'Power': pxx
    })
    return psd_df


def spectral_entropy(signal, fs):
    f, pxx = welch(signal, fs=fs)

    pxx_normalized = pxx / pxx.sum()

    entropy_value = entropy(pxx_normalized)

    entropy_df = pd.DataFrame({
        'Spectral Entropy': [entropy_value]
    })

    return entropy_df


def fft_features(signal, fs):
    n = len(signal)

    signal_fft = fft(signal)

    magnitudes = np.abs(signal_fft)

    freqs = np.fft.fftfreq(n, d=1/fs)

    half_n = n // 2

    fft_df = pd.DataFrame({
        'Frequency_fft': freqs[:half_n],
        'Magnitude': magnitudes[:half_n]
    })

    return fft_df


def measure_duration(segment, sampling_rate):
    peak_amplitude = np.max(segment)

    threshold = 0.5 * peak_amplitude

    crossing_points = np.where(segment > threshold)[0]

    if len(crossing_points) > 0:
        start_idx = crossing_points[0]
        end_idx = crossing_points[-1]
        duration_samples = end_idx - start_idx
        duration_seconds = duration_samples / sampling_rate
        return duration_seconds
    else:
        return 0

def calculate_advanced_temporal_features(signal, peaks, sampling_rate):
    window_size = int(0.05 * sampling_rate)

    durations = []

    amplitudes = []

    for peak in peaks:
        start = max(0, peak - window_size)
        end = min(len(signal), peak + window_size)
        segment = signal[start:end]

        # Duration calculation
        duration = measure_duration(segment, sampling_rate)
        durations.append(duration)

        # Amplitude calculation
        amplitude = np.max(segment) - np.min(segment)
        amplitudes.append(amplitude)

    # Create DataFrame from the duration and amplitude data
    features_df = pd.DataFrame({
        "durations": durations,
        "amplitudes": amplitudes
    })

    # Calculate aggregate statistics
    features_summary_df = pd.DataFrame({
        "average_duration": [features_df['durations'].mean()],
        "std_duration": [features_df['durations'].std()],
        "average_amplitude": [features_df['amplitudes'].mean()],
        "std_amplitude": [features_df['amplitudes'].std()]
    })

    return features_summary_df