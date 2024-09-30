from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import json
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict


def gradient_boosting_rhythm_classification_cv(combined_data):
    with open('best_params.json', 'r') as f:
        best_params = json.load(f)

    ecg_features = combined_data[['mean_heart_rate', 'mean_rr_interval', 'num_peaks', 'qt_interval_mean', 'pr_interval_mean','std_dev', 'max', 'min', 'kurtosis', 'mean'

                                 ,'auc'
                                 , 'SDNN','RMSSD', 'pNN50', 'HF_power',
                                 'Spectral Entropy',
                                 'Frequency_fft',
                                 'Magnitude',
                                 'average_amplitude',
                                 'std_amplitude',
                                 'average_duration'
                                 ]]

    imputer = SimpleImputer(strategy='median')
    features_imputed = imputer.fit_transform(ecg_features)

    y = combined_data['Rhythm']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

     # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_imputed)

     # Initialize cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = defaultdict(list)

    for train_index, test_index in skf.split(features_scaled, y_encoded):
        # Split data
        X_train, X_test = features_scaled[train_index], features_scaled[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        # Apply SMOTE
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # Initialize XGBClassifier with best parameters
        xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', **best_params)

        # Train the model
        xgb_classifier.fit(X_train_smote, y_train_smote)

        # Make predictions
        predictions = xgb_classifier.predict(X_test)

        # Calculate scores
        scores['accuracy'].append(accuracy_score(y_test, predictions))
        scores['precision'].append(precision_score(y_test, predictions, average='weighted', zero_division=0))
        scores['recall'].append(recall_score(y_test, predictions, average='weighted'))
        scores['f1'].append(f1_score(y_test, predictions, average='weighted'))

    # Convert scores dictionary to a DataFrame for easy viewing
    scores_df = pd.DataFrame(scores)

    return scores_df.mean(), scores_df.std()  # Return the mean and standard deviation of the scores

# Call the function with the combined data
combined_data = pd.read_csv("trainingModel/Gradient_Boost/combined_data_extra_features_3.csv")
mean_scores, std_scores = gradient_boosting_rhythm_classification_cv(combined_data)

print("Mean Scores:", mean_scores)
print("Standard Deviations:", std_scores)
