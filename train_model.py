"""
train_model.py — Train Linear Regression (baseline) and Random Forest (main model)
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import load_or_generate_data, preprocess


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  MAE  : ₹ {mae:,.0f}")
    print(f"  RMSE : ₹ {rmse:,.0f}")
    print(f"  R²   : {r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_and_save():
    print("Loading and preprocessing data...")
    df = load_or_generate_data()
    X, y, _, feature_cols, _ = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Linear Regression (baseline) ──────────────────────────────────────
    print("\nTraining Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_metrics = evaluate("Linear Regression", lr, X_test, y_test)

    # ── Random Forest (main model) ─────────────────────────────────────────
    print("\nTraining Random Forest Regressor...")
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_metrics = evaluate("Random Forest Regressor", rf, X_test, y_test)

    # Feature importances
    importances = dict(zip(feature_cols, rf.feature_importances_))
    importances_sorted = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    # Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(lr, "models/linear_regression.pkl")
    joblib.dump(rf, "models/random_forest.pkl")
    joblib.dump(importances_sorted, "models/feature_importances.pkl")
    joblib.dump({
        "lr": lr_metrics, "rf": rf_metrics,
        "X_test": X_test, "y_test": y_test
    }, "models/eval_data.pkl")

    print("\n✅ Models saved to models/")
    print("\nFeature Importances (RF):")
    for feat, imp in list(importances_sorted.items())[:8]:
        bar = "█" * int(imp * 50)
        print(f"  {feat:<20} {bar} {imp:.3f}")

    return lr, rf, rf_metrics


if __name__ == "__main__":
    train_and_save()
