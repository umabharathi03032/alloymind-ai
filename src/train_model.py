"""
train_model.py
---------------
Loads the alloy readings dataset, trains and compares two regression models
to predict the required composition adjustment, evaluates them with MAE/RMSE
on a held-out test set, and saves the best-performing model to disk.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ELEMENTS = ["C", "Si", "Mn", "P", "S", "Cr", "Ni"]
TARGET_COLS = [f"adj_{el}" for el in ELEMENTS]

import os
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_BASE_DIR, "data", "alloy_readings.csv")
MODEL_PATH = os.path.join(_BASE_DIR, "models", "alloy_model.pkl")


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[ELEMENTS].values
    y = df[TARGET_COLS].values
    return X, y


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"{name:20s} | MAE: {mae:.5f}  RMSE: {rmse:.5f}")
    return mae, rmse


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set size: {len(X_train)}  Test set size: {len(X_test)}\n")
    print("Model comparison (lower is better):")

    results = {}

    # Model 1: Linear Regression (simple baseline)
    lr = MultiOutputRegressor(LinearRegression())
    lr.fit(X_train, y_train)
    results["LinearRegression"] = (lr, *evaluate("LinearRegression", lr, X_test, y_test))

    # Model 2: Random Forest (chosen for speed + interpretability)
    rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42))
    rf.fit(X_train, y_train)
    results["RandomForest"] = (rf, *evaluate("RandomForest", rf, X_test, y_test))

    # Pick best model by RMSE
    best_name = min(results, key=lambda k: results[k][2])
    best_model = results[best_name][0]
    print(f"\nBest model: {best_name}")

    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best model -> {MODEL_PATH}")

    # Feature importance (only meaningful for the Random Forest)
    if best_name == "RandomForest":
        print("\nFeature importances (averaged across all 7 output targets):")
        importances = np.mean(
            [est.feature_importances_ for est in best_model.estimators_], axis=0
        )
        for el, imp in sorted(zip(ELEMENTS, importances), key=lambda x: -x[1]):
            print(f"  {el:4s}: {imp:.4f}")


if __name__ == "__main__":
    main()
