import matlab.engine
import pandas as pd
import numpy as np

def denoise_ecg_with_matlab(ecg_data):
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()

    eng.addpath(r'trainingModel/ECGDenois', nargout=0)
    # Convert DataFrame to MATLAB matrix
    ecg_matrix = matlab.double(ecg_data.values.tolist())

    denoised_ecg_matrix = eng.ECGDenoisingDirect(ecg_matrix, nargout=1)

    # Convert MATLAB matrix back to DataFrame
    denoised_ecg_data = pd.DataFrame(np.array(denoised_ecg_matrix), columns=ecg_data.columns)

    # Stop the MATLAB engine
    eng.quit()

    return denoised_ecg_data
