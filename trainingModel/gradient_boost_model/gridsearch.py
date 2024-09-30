from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

combined_data = pd.read_csv("trainingModel/gradient_boost_model/combined_data_extra_features_3.csv")

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

y = combined_data['Rhythm']
record_names = combined_data['filename'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test, train_record_names, test_record_names = train_test_split(
    ecg_features, y_encoded, record_names, test_size=0.1, random_state=42
)


# Define the preprocessing and model pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
])

# Define the parameter grid
param_grid = {
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__subsample': [0.8, 1],
    'classifier__colsample_bytree': [0.8, 1],
    'classifier__reg_lambda': [1, 10],
    'classifier__reg_alpha': [0, 0.1, 1]
}

# Configure the GridSearchCV object
grid_cv = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)

# Fit the model
grid_cv.fit(X_train, y_train)

# Save best parameters to JSON
best_params = grid_cv.best_params_
with open('best_params.json', 'w') as f:
    json.dump(best_params, f)

# Print best parameters and accuracy
print(f"Best parameters found: {grid_cv.best_params_}")
print(f"Best cross-validated accuracy: {grid_cv.best_score_:.4f}")

# Use the best estimator to make predictions on the test set
best_classifier = grid_cv.best_estimator_
predictions = best_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

# Output the performance metrics
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))


# Save to a JSON file
with open('best_params.json', 'w') as f:
    json.dump(best_params, f)


# Best parameters found: {'classifier__colsample_bytree': 0.8, 'classifier__learning_rate': 0.1, 'classifier__max_depth': 7, 'classifier__n_estimators': 300, 'classifier__reg_alpha': 0, 'classifier__reg_lambda': 1, 'classifier__subsample': 0.8}