% ECGDenoisingDirect - Denoises ECG data provided as a MATLAB matrix.
% DataFile - A MATLAB matrix where each column represents an ECG lead.
function DenoisingData = ECGDenoisingDirect(DataFile)
    [rows, ~] = size(DataFile);
    DenoisingData = zeros(rows, 12);

    % Processing each column (assuming 12 channels of ECG data)
    for j = 1:12
        OrigECG = DataFile(:, j);
        Fs = 500; % Sampling frequency
        fp = 50; fs = 60; % Passband and stopband frequencies
        rp = 1; rs = 2.5; % Passband ripple and stopband attenuation
        wp = fp / (Fs / 2); ws = fs / (Fs / 2);
        [n, wn] = buttord(wp, ws, rp, rs);
        [bz, az] = butter(n, wn);
        LPassDataFile = filtfilt(bz, az, OrigECG);

        % Smoothing and baseline wandering removal
        t = 1:length(LPassDataFile);
        yy2 = smooth(t, LPassDataFile, 0.1, 'rloess');
        BWRemoveDataFile = (LPassDataFile - yy2);

        % Differential filter for noise estimation
        Dl1 = BWRemoveDataFile;
        for k = 2:length(Dl1) - 1
            Dl1(k) = (2 * Dl1(k) - Dl1(k - 1) - Dl1(k + 1)) / sqrt(6);
        end
        NoisSTD = 1.4826 * median(abs(Dl1 - median(Dl1)));

        % Applying Non-Local Means Denoising
        % NOTE: Ensure the function NLM_1dDarbon is defined or appropriately included
        DenoisingData(:, j) = NLM_1dDarbon(BWRemoveDataFile, 1.5 * NoisSTD, 5000, 10);
    end

    % Note that we no longer save the data to a file, but return the matrix directly
    disp('Denoising completed.');
end
