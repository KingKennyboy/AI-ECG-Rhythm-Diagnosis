from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.impute import SimpleImputer


combined_data = pd.read_csv("trainingModel/gradient_boost_model/combined_data_extra_features_3.csv")
label_encoder = LabelEncoder()

# Create an imputer object with a mean filling strategy
imputer = SimpleImputer(strategy='mean')

# Assuming 'combined_data' is your final DataFrame after merging with labels
X = combined_data.drop(columns=['filename', 'FileName', 'Rhythm'])  # Feature matrix

X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

y = combined_data['Rhythm']  # Target variable
y_encoded = label_encoder.fit_transform(y)


# Initialize the RFE selector with cross-validation
rfe_selector = RFECV(estimator=XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), step=1, cv=3, scoring='accuracy')
rfe_selector.fit(X_imputed, y_encoded )

# Get which features are selected by RFE
rfe_selected = pd.DataFrame({
    'Feature': X.columns,
    'Selected': rfe_selector.support_,
    'Ranking': rfe_selector.ranking_
})

print(rfe_selected[rfe_selected['Selected']])