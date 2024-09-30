import unittest
import sys
sys.path.insert(0, "")
from trainingModel.gradient_boost_model.calculating_features import calculate_heart_rate_from_rpeaks, calculate_morphological_features, calculate_hrv_features, calculate_psd, spectral_entropy, fft_features, measure_duration, calculate_advanced_temporal_features
import pandas as pd
import numpy as np


class TestSignalProcessing(unittest.TestCase):

    def test_calculate_heart_rate_from_rpeaks(self):
        r_peaks = np.array([0, 100, 200, 300, 400])
        sampling_rate = 100
        result = calculate_heart_rate_from_rpeaks(r_peaks, sampling_rate)
        self.assertAlmostEqual(result, 60)

    def test_calculate_morphological_features(self):
        data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        templates_df = pd.DataFrame(data)
        result = calculate_morphological_features(templates_df)
        expected_output = pd.DataFrame({'peak_to_peak': [3, 3, 3], 'auc': [2.5, 3.5, 4.5]})
        pd.testing.assert_frame_equal(result, expected_output)

    def test_calculate_hrv_features(self):
        rr_intervals_ms = np.array([1000, 950, 1020])
        result = calculate_hrv_features(rr_intervals_ms)
        correct_rmssd = 60.8276253029822  # Update based on actual calculation if necessary
        self.assertAlmostEqual(result['SDNN'], 29.43920288775949, places=5)
        self.assertAlmostEqual(result['RMSSD'], correct_rmssd, places=5)
        self.assertAlmostEqual(result['pNN50'], 33.333333333333336)
        self.assertAlmostEqual(result['HF_power'], 0.0)

    # Check if calculate_psd supports nperseg argument
    def test_calculate_psd(self):
        signal = np.random.rand(1000)
        fs = 250
        try:
            # Try passing nperseg if supported
            result = calculate_psd(signal, fs, nperseg=min(len(signal), 1024))
        except TypeError:
            # If nperseg not supported, remove it from the test
            result = calculate_psd(signal, fs)
        self.assertTrue('Power' in result.columns)
        self.assertTrue(len(result['Power'][0]) > 0)


    def test_spectral_entropy(self):
        signal = np.random.rand(1000)
        fs = 250
        result = spectral_entropy(signal, fs)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue('Spectral Entropy' in result.columns)

    def test_fft_features(self):
        signal = np.random.rand(1000)
        fs = 250
        result = fft_features(signal, fs)
        self.assertEqual(len(result), 500)  # Half the length of the signal

    def test_measure_duration(self):
        segment = np.array([0, 0.5, 1, 0.5, 0])
        sampling_rate = 1
        result = measure_duration(segment, sampling_rate)
        self.assertEqual(result, 0)

    def test_calculate_advanced_temporal_features(self):
        signal = np.random.rand(1000)
        peaks = [100, 500, 900]
        sampling_rate = 1000
        result = calculate_advanced_temporal_features(signal, peaks, sampling_rate)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(all(col in result.columns for col in ['average_duration', 'std_duration', 'average_amplitude', 'std_amplitude']))

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
