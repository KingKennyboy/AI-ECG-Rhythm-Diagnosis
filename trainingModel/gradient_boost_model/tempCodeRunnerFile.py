import pandas as pd
import sys
from biosppy.signals import ecg
sys.path.insert(0, "")

from trainingModel.test_extract_features.function.calculating_features import calculate_heart_rate_from_rpeaks, calculate_qrs_duration

# Load ECG data from CSV
file_path = 'new DataSet/DenoisedDataSet/MUSE_20180209_122155_49000.csv'  # Update this with the actual file path
ecg_data = pd.read_csv(file_path, header=None)  # Assuming no header in the file

sampling_rate = 500
heart_rates = []
qrs_durations = []

# Iterate over each lead in the ECG data
for column in ecg_data.columns:
    # Extract the signal for the current lead
    ecg_signal = ecg_data[column].values

    # Analyze the ECG signal using biosppy
    out = ecg.ecg(signal=ecg_signal, sampling_rate=sampling_rate, show=False)

    # Calculate heart rate from R-peaks for this lead
    r_peaks = out['rpeaks']
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ## # # #
    # Calculate heart rate using the function
    heart_rate_bpm = calculate_heart_rate_from_rpeaks(r_peaks, sampling_rate)
    heart_rates.append(heart_rate_bpm)

    # Calculate QRS duration using the function
    qrs_duration = calculate_qrs_duration(ecg_signal, r_peaks,sampling_rate)
    qrs_durations.append(qrs_duration)



# Compute the average heart rate across all leads
average_heart_rate = sum(heart_rates) / len(heart_rates)
print("Average Heart Rate (BPM) across all leads:", average_heart_rate)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Compute the average QRS duration across all leads
average_qrs_duration = sum(qrs_durations) / len(qrs_durations)
print(f"Average QRS Duration (seconds) across all leads: {average_qrs_duration}")