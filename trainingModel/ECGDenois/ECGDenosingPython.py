import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def NLM_1dDarbon(signal, lambda_val, P, PatchHW):

    if isinstance(P, (int, float)):
        Pvec = np.arange(-P, P + 1)
    else:
        Pvec = P

    debug = {}
    N = len(signal)

    denoised_sig = np.full_like(signal, np.nan)

    # Don't bother denoising edges
    iStart = PatchHW + 1
    iEnd = N - PatchHW
    denoised_sig[iStart:iEnd] = 0

    debug['iStart'] = iStart
    debug['iEnd'] = iEnd

    # Initialize weight normalization
    Z = np.zeros_like(signal)
    cnt = np.zeros_like(signal)

    # Convert lambda value to 'h', denominator, as in original Buades papers
    Npatch = 2 * PatchHW + 1
    h = 2 * Npatch * lambda_val ** 2

    for idx in Pvec:  # Loop over all possible differences: s-t
        # Do summation over p - Eq. 3 in Darbon
        k = np.arange(N)
        kplus = k + idx
        igood = (kplus > 0) & (kplus <= N)  # Ignore OOB data; we could also handle it
        SSD = np.zeros_like(k)
        SSD[igood] = (signal[k[igood]] - signal[kplus[igood]]) ** 2
        Sdx = np.cumsum(SSD)

        for ii in range(iStart, iEnd):  # Loop over all points 's'
            distance = Sdx[ii + PatchHW] - Sdx[ii - PatchHW - 1]  # Eq 4; this is in place of point-by-point MSE

            w = np.exp(-distance / h)  # Eq 2 in Darbon
            t = ii + idx  # In the papers, this is not made explicit

            if 1 <= t <= N:
                denoised_sig[ii] += w * signal[t]
                Z[ii] += w
                cnt[ii] += 1

    # Apply normalization
    denoised_sig /= (Z + 1e-8)
    denoised_sig[:PatchHW + 1] = signal[:PatchHW + 1]
    denoised_sig[-PatchHW:] = signal[-PatchHW:]
    debug['Z'] = Z

    return denoised_sig, debug

def ecg_denoising(ecg_data, fs=500):

    rows, cols = ecg_data.shape
    denoised_data = np.zeros((rows, cols))

    # Processing each column (assuming 12 channels of ECG data)
    for j in range(cols):
        orig_ecg = ecg_data.iloc[:, j]
        fp = 50  # Passband frequency
        fsb = 60  # Stopband frequency
        rp = 1  # Passband ripple
        rs = 2.5  # Stopband attenuation
        wp = fp / (fs / 2)  # Normalize the frequency
        ws = fsb / (fs / 2)  # Normalize the frequency
        n, wn = butter(1, [wp, ws], btype='band', analog=False)
        b, a = butter(n, wn, btype='band', analog=False)
        low_pass_data = filtfilt(b, a, orig_ecg)

        # Smoothing and baseline wandering removal
        window_length = int(0.1 * len(low_pass_data))
        yy2 = pd.Series(low_pass_data).rolling(window=window_length, min_periods=1, center=True).mean().values
        bw_remove_data = low_pass_data - yy2

        # Differential filter for noise estimation
        dl1 = np.copy(bw_remove_data)
        for k in range(1, len(dl1) - 1):
            dl1[k] = (2 * dl1[k] - dl1[k - 1] - dl1[k + 1]) / np.sqrt(6)
        noise_std = 1.4826 * np.median(np.abs(dl1 - np.median(dl1)))

        # Applying Non-Local Means Denoising
        denoised_data[:, j] = NLM_1dDarbon(bw_remove_data, 1.5 * noise_std, 5000, 10)

    # Convert the denoised data to a DataFrame and return it
    denoised_data_df = pd.DataFrame(denoised_data, columns=ecg_data.columns)
    return denoised_data_df

