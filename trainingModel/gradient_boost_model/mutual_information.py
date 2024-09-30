from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.impute import SimpleImputer


combined_data = pd.read_csv("trainingModel/gradient_boost_model/combined_data_extra_features_3.csv")
label_encoder = LabelEncoder()


imputer = SimpleImputer(strategy='mean')

X = combined_data.drop(columns=['filename', 'FileName', 'Rhythm'])

X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

y = combined_data['Rhythm']  # Target variable
y_encoded = label_encoder.fit_transform(y)

# Apply Mutual Information Feature Selection
mi_selector = SelectKBest(mutual_info_classif, k='all')
mi_selector.fit(X_imputed, y_encoded)

# Get MI scores for all features
mi_scores = pd.DataFrame({
    'Feature': X.columns,
    'MI_Score': mi_selector.scores_
}).sort_values(by='MI_Score', ascending=False)

print(mi_scores)