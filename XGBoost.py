import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
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

# -----------------------------
# 4. Hyperparameter Tuning with Optuna
# -----------------------------
def objective(trial):
    param = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "tree_method": "hist",       
        "device": "cuda" 
    }
    
    # Use the custom classifier
    model = xgb.XGBClassifier(**param, use_label_encoder=False, eval_metric='mlogloss')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X, y, cv=kf, scoring='accuracy').mean()
    
    return score

# Create and run the Optuna study
print("Launching optuna study...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print("Best hyperparameters: ", study.best_trial.params)

best_params = study.best_trial.params

with open("best_params.json", "w") as f:
    json.dump(study.best_trial.params, f)

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
          early_stopping_rounds=10, 
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