"""
Shell.ai Hackathon - Material Blending Prediction Model
Score: 83.16 on public leaderboard | Execution Time: 21 minutes
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
import shap

# Config
RANDOM_STATE = 42
N_FOLDS = 5
N_SHAP_FEATURES = 50
N_TARGETS = 10

# Load
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Physics-inspired feature engineering
def engineer_features(df):
    """
    Engineer physics-inspired features for material blending prediction.
    Includes entropy, weighted properties, and component ratios.
    """
    vol_cols = [f'Component{i}_fraction' for i in range(1, 6)]
    volumes = df[vol_cols].values
    df['VolumeEntropy'] = -np.sum(volumes * np.log(volumes + 1e-10), axis=1)
    df['VolumeStd'] = df[vol_cols].std(axis=1)

    for prop in range(1, 11):
        prop_cols = [f'Component{i}_Property{prop}' for i in range(1, 6)]
        props = df[prop_cols].values
        df[f'WeightedProp{prop}'] = np.sum(volumes * props, axis=1)
        df[f'MinProp{prop}'] = np.min(props, axis=1)
        df[f'MaxProp{prop}'] = np.max(props, axis=1)
        df[f'RangeProp{prop}'] = df[f'MaxProp{prop}'] - df[f'MinProp{prop}']
        df[f'LogProp{prop}'] = np.exp(np.sum(volumes * np.log(props + 1e-10), axis=1))
        df[f'HarmonicProp{prop}'] = 1 / (np.sum(volumes / (props + 1e-10), axis=1) + 1e-10)

        for i in range(1, 6):
            df[f'Vol{i}_Prop{prop}'] = df[f'Component{i}_fraction'] * df[f'Component{i}_Property{prop}']

    df['Comp1_Comp5_ratio'] = df['Component1_fraction'] / (df['Component5_fraction'] + 1e-10)
    df['Comp3_vol_prop8'] = df['Component3_fraction'] * df['Component3_Property8']
    return df

# Engineer
train = engineer_features(train)
test = engineer_features(test)

# Features and Targets
X = train.drop(columns=[f'BlendProperty{i}' for i in range(1, 11)])
y = train[[f'BlendProperty{i}' for i in range(1, 11)]]
X_test = test.copy()

# LGB params
lgb_params = {
    'objective': 'regression_l1',
    'metric': 'mape',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'random_state': RANDOM_STATE,
    'n_estimators': 2000,
    'verbosity': -1
}

# MAPE metric
def competition_mape(y_true, y_pred, eps=1e-10):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs(y_true - y_pred) / np.maximum(eps, np.abs(y_true)))

# Store predictions
test_predictions = np.zeros((len(X_test), N_TARGETS))

for i in range(1, N_TARGETS + 1):
    target = f'BlendProperty{i}'
    print(f"\n🔧 Processing target: {target}")

    # SHAP-based feature selection
    base_model = lgb.LGBMRegressor(**lgb_params)
    base_model.fit(X, y[target])
    explainer = shap.TreeExplainer(base_model)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_mean = np.abs(shap_vals).mean(axis=0)
    top_indices = np.argsort(shap_mean)[-N_SHAP_FEATURES:]
    selected_features = X.columns[top_indices]

    # Train with cross-validation
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, y_train = X.iloc[train_idx][selected_features], y[target].iloc[train_idx]
        X_val, y_val = X.iloc[val_idx][selected_features], y[target].iloc[val_idx]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mape',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(False)
            ]
        )
        preds += model.predict(X_test[selected_features]) / N_FOLDS

    # Physics-based clipping
    preds = np.clip(preds, y[target].min(), y[target].max())
    test_predictions[:, i - 1] = preds
    print(f"✅ Completed target {i} with clipping range: [{y[target].min():.4f}, {y[target].max():.4f}]")

# Submission
submission = pd.DataFrame(test_predictions, columns=[f'BlendProperty{i}' for i in range(1, 11)])
submission.insert(0, 'ID', test.index)
submission.to_csv('outputs/submission.csv', index=False)
print("\n✅ Submission saved as outputs/submission.csv")
