"""
train.py - CRISPR-Cas9 On-Target Efficiency LightGBM Regressor Trainer

Workflow:
1. Loads benchmark dataset (doench_benchmark.csv).
2. Extracts biophysical & positional features using features.py.
3. Performs 80/20 train-test split (reproducible seed=42).
4. Trains LightGBM Regressor (n_estimators=150, learning_rate=0.05).
5. Evaluates on test set using Spearman Rank Correlation (standard CRISPR benchmark metric),
   Pearson r, RMSE, and R2.
6. Analyzes top predictive feature importances.
7. Serializes the trained model to 'crispr_lgbm_model.pkl' using joblib.
"""

import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Ensure local module import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import extract_features_df, get_feature_names


def train_model(
    data_path: str = "doench_benchmark.csv",
    model_output_path: str = "crispr_lgbm_model.pkl",
    n_estimators: int = 150,
    learning_rate: float = 0.05,
    test_size: float = 0.20,
    random_state: int = 42,
):
    print("=" * 60)
    print(" CRISPR-Cas9 ON-TARGET EFFICIENCY MODEL TRAINING ")
    print("=" * 60)

    # 1. Load Data
    data_file = os.path.abspath(data_path)
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Dataset not found at {data_file}. Run get_data.py first to generate benchmark data."
        )

    print(f"[+] Loading dataset from: {data_file}")
    df = pd.read_csv(data_file)
    print(f"    - Total samples: {len(df)}")
    print(f"    - Target variable ('activity') summary:")
    print(f"      Mean: {df['activity'].mean():.4f} | Std: {df['activity'].std():.4f} | Min: {df['activity'].min():.4f} | Max: {df['activity'].max():.4f}")

    # 2. Feature Extraction
    print(f"[+] Extracting features for {len(df)} 23-mer sequences...")
    X = extract_features_df(df["sequence"])
    y = df["activity"].values
    feature_names = get_feature_names()
    print(f"    - Extracted feature matrix shape: {X.shape} ({X.shape[1]} features)")

    # 3. Train / Test Split (80/20)
    print(f"[+] Performing {int((1 - test_size) * 100)}/{int(test_size * 100)} train/test split (seed={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"    - Train set size: {len(X_train)} samples")
    print(f"    - Test set size:  {len(X_test)} samples")

    # 4. LightGBM Model Initialization & Training
    print(f"[+] Initializing LightGBM Regressor (n_estimators={n_estimators}, learning_rate={learning_rate})...")
    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
        verbose=-1,
        n_jobs=-1,
    )

    print("[+] Fitting model on training split...")
    model.fit(X_train, y_train)

    # 5. Model Evaluation
    print("[+] Evaluating on unseen test set...")
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Spearman Rank Correlation (standard benchmark metric for CRISPR gRNA ranking)
    spearman_rho, spearman_p = spearmanr(y_test, y_pred_test)
    pearson_r, pearson_p = pearsonr(y_test, y_pred_test)

    # Error metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    train_spearman, _ = spearmanr(y_train, y_pred_train)

    print("\n" + "-" * 40)
    print(" MODEL PERFORMANCE EVALUATION (TEST SET)")
    print("-" * 40)
    print(f"  Spearman Rank Correlation (rho): {spearman_rho:.4f} (p = {spearman_p:.2e})")
    print(f"  Pearson Correlation (r):         {pearson_r:.4f} (p = {pearson_p:.2e})")
    print(f"  Coefficient of Det. (R^2):       {test_r2:.4f}")
    print(f"  Root Mean Squared Error (RMSE):  {test_rmse:.4f}")
    print(f"  Mean Absolute Error (MAE):       {test_mae:.4f}")
    print(f"  Train Spearman rho:              {train_spearman:.4f}")
    print("-" * 40)

    # 6. Feature Importance Analysis
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    print("\n[+] Top 10 Most Influential Features:")
    for rank, (_, row) in enumerate(feat_imp_df.head(10).iterrows(), start=1):
        print(f"    {rank:2d}. {row['feature']:<15} (Importance: {row['importance']})")

    # 7. Model Serialization
    output_file = os.path.abspath(model_output_path)
    joblib.dump(model, output_file)
    print(f"\n[OK] Trained model successfully saved to: {output_file}")

    # 8. Verification of serialized artifact
    loaded_model = joblib.load(output_file)
    sample_pred = loaded_model.predict(X_test.iloc[:1])
    print(f"[OK] Serialization verified. Sample prediction on test[0]: {sample_pred[0]:.4f} (True: {y_test[0]:.4f})")

    return model, spearman_rho


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM on-target CRISPR guide predictor.")
    parser.add_argument("--data", type=str, default="doench_benchmark.csv", help="Path to input dataset CSV")
    parser.add_argument("--output", type=str, default="crispr_lgbm_model.pkl", help="Path to save trained model")
    parser.add_argument("--n_estimators", type=int, default=150, help="Number of estimators (default: 150)")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate (default: 0.05)")
    args = parser.parse_args()

    train_model(
        data_path=args.data,
        model_output_path=args.output,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
