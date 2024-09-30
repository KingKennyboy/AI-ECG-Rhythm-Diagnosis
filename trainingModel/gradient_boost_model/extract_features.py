import pandas as pd
import numpy as np
import neurokit2 as nk
from biosppy.signals import ecg
import sys
import scipy.stats
sys.path.insert(0, "")
# Assuming your module paths and imports are set correctly
from trainingModel.gradient_boost_model.calculating_features import calculate_advanced_temporal_features, calculate_heart_rate_from_rpeaks, calculate_hrv_features, calculate_morphological_features, calculate_psd, fft_features, spectral_entropy

def process_ecg_data(ecg_data, sampling_rate=500):
    """Process ECG data from a DataFrame and extract features for each lead."""
    features_list = []

    # Process each lead in the ECG data
    for idx, lead_signal in ecg_data.items():
        if lead_signal.isnull().all():
            continue  # Skip leads that are entirely NaN

        # Clean the signal
        mean_value = lead_signal.mean()
        clean_lead_signal = lead_signal.fillna(mean_value).values

        # Ensure the signal has enough data points
        if len(clean_lead_signal) > 1:
            try:
                out = ecg.ecg(signal=clean_lead_signal, sampling_rate=sampling_rate, show=False)
                r_peaks = out['rpeaks']
                if len(r_peaks) < 2:  # Ensure there are enough peaks to calculate intervals
                    continue

                heart_rate = calculate_heart_rate_from_rpeaks(r_peaks, sampling_rate)
                rr_intervals = np.diff(r_peaks) / sampling_rate
                rr_intervals_ms = rr_intervals * 1000  # convert to milliseconds for HRV metrics

                # Calculate HRV features using the defined function
                hrv_features = calculate_hrv_features(rr_intervals_ms)
                hrv_features_df = pd.DataFrame([hrv_features])

                signal_processed, waves_dwt = nk.ecg_delineate(clean_lead_signal, r_peaks, sampling_rate=sampling_rate, method="dwt")
                # Extract intervals and peaks
                p_peaks = waves_dwt.get('ECG_P_Peaks', [])
                p_onsets = waves_dwt.get('ECG_P_Onsets', [])
                p_offsets = waves_dwt.get('ECG_P_Offsets', [])
                q_peaks = waves_dwt.get('ECG_Q_Peaks', [])
                r_onsets = waves_dwt.get('ECG_R_Onsets', [])
                r_offsets = waves_dwt.get('ECG_R_Offsets', [])
                s_peaks = waves_dwt.get('ECG_S_Peaks', [])
                t_peaks = waves_dwt.get('ECG_T_Peaks', [])
                t_onsets = waves_dwt.get('ECG_T_Onsets', [])
                t_offsets = waves_dwt.get('ECG_T_Offsets', [])

                # Compute PR and QT intervals if possible
                pr_intervals = [p_on - q for p_on, q in zip(p_onsets, q_peaks) if not np.isnan(p_on) and not np.isnan(q)]
                qt_intervals = [t_off - q for t_off, q in zip(t_offsets, q_peaks) if not np.isnan(t_off) and not np.isnan(q)]

                templates = out['templates']
                templates_df = pd.DataFrame(templates.T)
                features_df = pd.DataFrame({
                    'mean': templates_df.mean(axis=1),
                    'std_dev': templates_df.std(axis=1),
                    'max': templates_df.max(axis=1),
                    'min': templates_df.min(axis=1),
                    'skewness': templates_df.apply(lambda x: scipy.stats.skew(x), axis=1),
                    'kurtosis': templates_df.apply(lambda x: scipy.stats.kurtosis(x), axis=1)
                })
                # Averages down the columns

                morph_features = calculate_morphological_features(templates_df)
                fft_features_df = fft_features(clean_lead_signal, 500)
                psd_df = calculate_psd(clean_lead_signal, sampling_rate)
                entropy_df = spectral_entropy(clean_lead_signal, sampling_rate)
                temporal_features = calculate_advanced_temporal_features(clean_lead_signal, r_peaks, sampling_rate)
                features_df = pd.concat([features_df, morph_features, fft_features_df, hrv_features_df, psd_df, entropy_df, temporal_features], axis=1)

                average_features = features_df.mean(axis=0,numeric_only=True)

                # Combine all features into one dictionary
                features = {
                    'mean_heart_rate': heart_rate,
                    'mean_rr_interval': np.mean(rr_intervals) if len(rr_intervals) > 0 else np.nan,
                    'num_peaks': len(p_peaks),
                    'pr_interval_mean': np.mean(pr_intervals) if pr_intervals else np.nan,
                    'qt_interval_mean': np.mean(qt_intervals) if qt_intervals else np.nan,
                    **average_features.to_dict()  # This unpacks average_features and adds them to the features dictionary
                }
                features_list.append(features)

            except Exception as e:
                print(f"Error processing lead {idx} in {file_path}: {e}")

    # Compile features into a DataFrame
    if features_list:
        features_df = pd.DataFrame(features_list)
        return features_df
    return pd.DataFrame()  # Return empty DataFrame if no features were extracted

# Example usage
if __name__ == "__main__":
    file_path = 'trainingModel/ECGDataProcessed/MUSE_20180111_155203_15000.csv'
    ecg_data = pd.read_csv(file_path, header=None)  # Correctly load the ECG data
    features_df = process_ecg_data(ecg_data)  # Pass the DataFrame, not the file path
    print(features_df)