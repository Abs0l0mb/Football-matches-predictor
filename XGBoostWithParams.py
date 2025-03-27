import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import json


# -----------------------------
# 1. Load Datasets
# -----------------------------
print("Loading dataset...")
matches = pd.read_csv("Matches.csv")
elo = pd.read_csv("EloRatings.csv")

print(matches["HomeShots"])

# -----------------------------
# 2. Data Preprocessing & Completion
# -----------------------------
# Convert date columns to datetime format (if needed)
print("Encoding non-numeric data...")
matches['MatchDate'] = pd.to_datetime(matches['MatchDate'], errors='coerce')

# Encode categorical features (e.g., Division, HomeTeam, AwayTeam, FTResult, HTResult)
categorical_cols = ['Division', 'HomeTeam', 'AwayTeam', 'FTResult', 'HTResult']
for col in categorical_cols:
    if col in matches.columns:
        # Fill missing with a placeholder
        matches[col] = matches[col].fillna("Unknown")
        le = LabelEncoder()
        matches[col] = le.fit_transform(matches[col])

print("Imputing missing values...")
# Impute missing numeric values using regression-based imputation
numeric_cols = matches.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='median')
matches[numeric_cols] = imputer.fit_transform(matches[numeric_cols])
'''
numeric_cols = matches.select_dtypes(include=[np.number]).columns
imputer = IterativeImputer(random_state=42)
matches[numeric_cols] = imputer.fit_transform(matches[numeric_cols])
'''

# -----------------------------
# 3. Define Features and Target
# -----------------------------
# Here we drop columns that are not used for modeling such as dates or times.
# We use "FTResult" (the final match result) as our target.
X = matches.drop(columns=['FTResult', 'MatchDate', 'MatchTime', 'HTResult', 'FTHome', 'FTAway', 'HTHome', 'HTAway'], errors='ignore')
y = matches['FTResult']


with open("best_params.json", "r") as f:
    best_params = json.load(f)
    
# -----------------------------
# 5. Final Model Training with Early Stopping
# -----------------------------
# Split data into training and validation sets for early stopping
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, 
                                                  random_state=42, stratify=y)

# Initialize the final model using the best hyperparameters
model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='mlogloss')

# Train the model with an early stopping mechanism (auto-stopping after 10 rounds with no improvement)
model.fit(X_train, y_train, 
          eval_set=[(X_val, y_val)], 
          verbose=True)

# -----------------------------
# 6. Output Feature Importance Graph
# -----------------------------
# Plot and display the feature importance
xgb.plot_importance(model)
plt.title("Feature Importance")
plt.savefig("feature_importance.png")
plt.close()