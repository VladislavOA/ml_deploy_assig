import argparse
import json
import os
from catboost import CatBoostClassifier
import joblib
import mlflow
import mlflow.catboost
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


import sys

# Support running directly or via DVC/package
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from feature_engineering import engineer_features


def train_model(
    train_path: str,
    test_path: str,
    output_model_dir: str,
    iterations: int = 500,
    random_seed: int = 42,
    experiment_name: str = "churn-prediction",
):
    """
    Trains a CatBoostClassifier with balanced class weights,
    evaluates ROC-AUC and F1 on train and test datasets,
    logs parameters, metrics, and model to MLflow,
    and saves the model package to models/ directory.
    """
    print(f"Loading train data from: {train_path}")
    print(f"Loading test data from: {test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if "Churn" not in train_df.columns or "Churn" not in test_df.columns:
        raise ValueError("Target column 'Churn' not found in processed data")

    target_col = "Churn"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(int)

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(int)

    # Apply feature engineering (Stage 2)
    print("Applying feature engineering to train and test sets...")
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    # Detect categorical features (object, string, category)
    cat_features = list(
        X_train.select_dtypes(include=["object", "string", "category"]).columns
    )
    # Ensure categorical columns in both train and test are strings to avoid type issues
    for col in cat_features:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    print(f"Features count: {X_train.shape[1]}")
    print(f"Categorical features ({len(cat_features)}): {cat_features}")

    # Set up MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="catboost-balanced-weights") as run:
        print(f"MLflow Run ID: {run.info.run_id}")

        # Initialize CatBoost with balanced weights
        model = CatBoostClassifier(
            iterations=iterations,
            random_seed=random_seed,
            auto_class_weights="Balanced",
            eval_metric="F1",
            verbose=100,
        )

        print("\n--- Training CatBoostClassifier ---")
        model.fit(
            X_train,
            y_train,
            cat_features=cat_features,
            eval_set=(X_test, y_test),
            early_stopping_rounds=50,
            verbose=100,
        )

        # Predictions & Probabilities
        train_probs = model.predict_proba(X_train)[:, 1]
        train_preds = (train_probs >= 0.5).astype(int)

        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= 0.5).astype(int)

        # Compute Metrics
        train_roc_auc = float(roc_auc_score(y_train, train_probs))
        train_f1 = float(f1_score(y_train, train_preds))
        train_acc = float(accuracy_score(y_train, train_preds))
        train_prec = float(precision_score(y_train, train_preds))
        train_rec = float(recall_score(y_train, train_preds))

        test_roc_auc = float(roc_auc_score(y_test, test_probs))
        test_f1 = float(f1_score(y_test, test_preds))
        test_acc = float(accuracy_score(y_test, test_preds))
        test_prec = float(precision_score(y_test, test_preds))
        test_rec = float(recall_score(y_test, test_preds))

        # Report to console
        print("\n" + "=" * 50)
        print("           MODEL EVALUATION REPORT")
        print("=" * 50)
        print(f"Train ROC-AUC : {train_roc_auc:.4f}")
        print(f"Train F1-Score: {train_f1:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print("-" * 50)
        print(f"Test  ROC-AUC : {test_roc_auc:.4f}")
        print(f"Test  F1-Score: {test_f1:.4f}")
        print(f"Test  Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_prec:.4f}")
        print(f"Test  Recall  : {test_rec:.4f}")
        print("=" * 50)

        # Log parameters to MLflow
        mlflow.log_params(
            {
                "model_type": "CatBoostClassifier",
                "auto_class_weights": "Balanced",
                "iterations": iterations,
                "random_seed": random_seed,
                "best_iteration": model.get_best_iteration(),
                "n_features": X_train.shape[1],
                "n_cat_features": len(cat_features),
            }
        )

        # Log metrics to MLflow
        mlflow.log_metrics(
            {
                "train_roc_auc": train_roc_auc,
                "train_f1": train_f1,
                "train_accuracy": train_acc,
                "train_precision": train_prec,
                "train_recall": train_rec,
                "test_roc_auc": test_roc_auc,
                "test_f1": test_f1,
                "test_accuracy": test_acc,
                "test_precision": test_prec,
                "test_recall": test_rec,
            }
        )

        # Log model artifact to MLflow
        mlflow.catboost.log_model(model, name="model")
        print("Model and metrics successfully logged to MLflow.")

        # Save model locally in output_model_dir (models/)
        os.makedirs(output_model_dir, exist_ok=True)
        cbm_path = os.path.join(output_model_dir, "catboost_model.cbm")
        pkl_path = os.path.join(output_model_dir, "catboost_model.pkl")
        features_path = os.path.join(output_model_dir, "model_features.json")

        model.save_model(cbm_path)
        joblib.dump(model, pkl_path)

        # Save feature schema for API deployment
        feature_metadata = {
            "features": list(X_train.columns),
            "cat_features": cat_features,
            "numeric_features": [
                c for c in X_train.columns if c not in cat_features
            ],
            "metrics": {
                "test_roc_auc": test_roc_auc,
                "test_f1": test_f1,
                "train_roc_auc": train_roc_auc,
                "train_f1": train_f1,
            },
        }
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(feature_metadata, f, indent=2)

        print(f"Native model saved to: {cbm_path}")
        print(f"Pickle model saved to: {pkl_path}")
        print(f"Feature metadata saved to: {features_path}")

    return {
        "train_roc_auc": train_roc_auc,
        "train_f1": train_f1,
        "test_roc_auc": test_roc_auc,
        "test_f1": test_f1,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Stage with CatBoost & MLflow")
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/processed/train.csv",
        help="Path to processed training CSV",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed/test.csv",
        help="Path to processed testing CSV",
    )
    parser.add_argument(
        "--output-model-dir",
        type=str,
        default="models",
        help="Directory to save the trained model artifacts",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Number of CatBoost boosting iterations",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for CatBoost training",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="churn-prediction",
        help="MLflow experiment name",
    )
    args = parser.parse_args()

    train_model(
        train_path=args.train_data,
        test_path=args.test_data,
        output_model_dir=args.output_model_dir,
        iterations=args.iterations,
        random_seed=args.random_seed,
        experiment_name=args.experiment_name,
    )
