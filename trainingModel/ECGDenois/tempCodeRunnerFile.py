import matlab.engine
import pandas as pd
import numpy as np

def denoise_ecg_with_matlab(ecg_data):
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()

    # Convert DataFrame to MATLAB matrix
    ecg_matrix = matlab.double(ecg_data.values.tolist())

    denoised_ecg_matrix = eng.ECGDenoising(ecg_matrix, nargout=1)

    # Convert MATLAB matrix back to DataFrame
    denoised_ecg_data = pd.DataFrame(np.array(denoised_ecg_matrix), columns=ecg_data.columns)

    # Stop the MATLAB engine
    eng.quit()

    return denoised_ecg_data



# Example usage
if __name__ == "__main__":
    ecg_data = pd.read_csv("dataset/Original Dataset/MUSE_20180111_155115_19000.csv")  # Load your ECG data
    denoised_ecg_data = denoise_ecg_with_matlab(ecg_data)
    print(denoised_ecg_data.head())