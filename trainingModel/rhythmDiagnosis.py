import csv
import sys
import torch
import numpy as np
import pandas as pd
from io import StringIO
import joblib
from statistics import mode
import json
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "")
from trainingModel.ECGDenois.ECGDenoisingDirect import denoise_ecg_with_matlab
from trainingModel.timesnet_diff_method.main import args
from trainingModel.timesnet_diff_method.source_codes.Exp_Classification import Exp_Classification
from trainingModel.gradient_boost_model.extract_features import process_ecg_data
from trainingModel.timesnet_diff_method.check_input_data import check_csv_columns


input_data = sys.stdin.read()
data = StringIO(input_data)
ecg_data = pd.read_csv(data)

message = ''
model_type = sys.argv[2]
print(f"Using model type: {model_type}")

csv_data = csv.reader(input_data.splitlines())

if not check_csv_columns(csv_data):
    message = "Error: ECG does not contain 12 leads"

if message == '' and np.isnan(ecg_data.values).any():
    message = 'Error: Corrupted ECG Signal, contains null values'


if message == '':
    if model_type == "TimeSnet Model":
        label_classes = ['AFIB', 'AT', 'SA', 'SND', 'SR']
        print(f"Using model type: {model_type}")
        ecg_data = np.array(list(csv_data), dtype=np.float32)
        mean, std = 23.21351997, 118.824138
        normalized_data = (ecg_data - mean) / std
        tensor = torch.tensor(normalized_data, dtype=torch.float32)
        exp = Exp_Classification(args)
        rhythm_index = exp.single_data_test(tensor)
        rhythm_output = label_classes[rhythm_index]
        print(rhythm_output)

    elif model_type == "ECG XGBoost Model":
        model = joblib.load('trainingModel/gradient_boost_model/xgb_classifier.pkl')
        label_encoder = joblib.load('trainingModel/gradient_boost_model/label_encoder.pkl')
        imputer = joblib.load('trainingModel/gradient_boost_model/imputer.pkl')
        scaler = joblib.load('trainingModel/gradient_boost_model/scaler.pkl')
        feature_names = joblib.load('trainingModel/gradient_boost_model/feature_names.pkl')

        ecg_data_denoised = denoise_ecg_with_matlab(ecg_data)
        features_df = process_ecg_data(ecg_data_denoised, sampling_rate=500)
        features_df = features_df[feature_names]

        features_df.drop(columns=['lead_index'], errors='ignore', inplace=True)
        features_df = features_df.apply(lambda col: col.astype('category') if col.dtype == 'object' else col)


        features_imputed = imputer.transform(features_df)
        features_scaled = scaler.transform(features_imputed)

        predictions = model.predict(features_scaled)
        readable_predictions = label_encoder.inverse_transform(predictions)
        try:
            rhythm_output = mode(readable_predictions)
        except Exception as e:
            print("Error in finding the most common rhythm:", e)
else:

    print(message)
    rhythm_output = None

output_data = {
    "message": message,
    "diagnosis_result": rhythm_output,
    "model": model_type,
    "patient_name": sys.argv[1],
    # "filename": base_filename,  # temporary code for presentation
    # "corresponding_rhythm": corresponding_rhythm  # temporary code
}

print('---JSON_START---')
print(json.dumps(output_data))
print('---JSON_END---')
