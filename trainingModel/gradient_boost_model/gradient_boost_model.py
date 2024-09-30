import pandas as pd
import joblib
import json
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def gradient_boosting_rhythm_classification(combined_data):
    # Load best parameters for the XGBoost classifier
    with open('best_params.json', 'r') as f:
        best_params = json.load(f)

    # Define feature columns to be used
    feature_columns = [
        'mean_heart_rate', 'mean_rr_interval', 'num_peaks', 'qt_interval_mean', 'pr_interval_mean',
        'std_dev', 'max', 'min', 'kurtosis', 'mean', 'auc', 'SDNN', 'RMSSD', 'pNN50', 'HF_power',
        'peak_to_peak', 'skewness', 'Spectral Entropy', 'Frequency_fft', 'Magnitude',
        'average_amplitude', 'std_amplitude', 'average_duration', 'std_duration'
    ]

    # Prepare features and target
    features = combined_data[feature_columns]
    y = combined_data['Rhythm']
    filenames = combined_data['filename']  # Capture filenames for reference

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data into training and test sets while preserving indices
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
        features, y_encoded, filenames, test_size=0.1, random_state=42, stratify=y_encoded
    )

    # Handle missing values and scale data
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)
    feature_names = X_train.columns.tolist()

    # Initialize and train the XGBoost classifier
    xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric=['mlogloss', 'merror'], **best_params)
    xgb_classifier.fit(X_train_scaled, y_train, eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)], verbose=True)

    # Predict on test data
    predictions = xgb_classifier.predict(X_test_scaled)
    predicted_rhythms = label_encoder.inverse_transform(predictions)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    # Compile results into a DataFrame
    predictions_df = pd.DataFrame({
        'Record Name': filenames_test,
        'Actual Rhythm': label_encoder.inverse_transform(y_test),
        'Predicted Rhythm': predicted_rhythms,
        'Match': label_encoder.inverse_transform(y_test) == predicted_rhythms
    })

    # Output evaluation metrics
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Save the trained model, label encoder, imputer, and scaler
    joblib.dump(xgb_classifier, 'trainingModel/gradient_boost_model/xgb_classifier.pkl')
    joblib.dump(label_encoder, 'trainingModel/gradient_boost_model/label_encoder.pkl')
    joblib.dump(imputer, 'trainingModel/gradient_boost_model/imputer.pkl')
    joblib.dump(scaler, 'trainingModel/gradient_boost_model/scaler.pkl')
    joblib.dump(feature_names, 'trainingModel/gradient_boost_model/feature_names.pkl')

    return predictions_df, xgb_classifier, label_encoder


# Example usage
combined_data = pd.read_csv("trainingModel/gradient_boost_model/combined_data.csv")
predictions_df, xgb_classifier, label_encoder = gradient_boosting_rhythm_classification(combined_data)
print(predictions_df)
# Filter to find mismatches
mismatches = predictions_df[predictions_df['Match'] == False]
matches = predictions_df[predictions_df['Match'] == True]
mismatch_count = (predictions_df['Match'] == False).sum()

matches.to_csv("trainingModel/gradient_boost_model/matches_test.csv")
mismatches.to_csv("trainingModel/gradient_boost_model/mismatches_test.csv")
# Display mismatches
print("Mismatches found:")
print(mismatches)
print(f"Number of mismatches: {mismatch_count}")

# Analyze the most frequently mispredicted rhythms
mispredicted_counts = mismatches['Actual Rhythm'].value_counts()
print("Frequency of Mispredicted Rhythms:")
print(mispredicted_counts)

# Cross tabulation of actual vs predicted rhythms in mismatches
confusion_mismatches = pd.crosstab(mismatches['Actual Rhythm'], mismatches['Predicted Rhythm'])
print("Confusion Matrix for Mismatches:")
print(confusion_mismatches)


