"""
MLProject/modelling.py
Script training Breast Cancer dengan argparse untuk MLflow Project.
"""

import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os, json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth",    type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    train = pd.read_csv("../breast_cancer_preprocessing/breast_cancer_train.csv")
    test = pd.read_csv("../breast_cancer_preprocessing/breast_cancer_test.csv")
    feature_cols = [c for c in train.columns if c != 'target']
    X_train = train[feature_cols]; y_train = train['target']
    X_test  = test[feature_cols];  y_test  = test['target']
    labels  = ['malignant', 'benign']

    mlflow.set_experiment("breast-cancer-ci")

    
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth",    args.max_depth)
    mlflow.log_param("random_state", args.random_state)
    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall",    rec)
    mlflow.log_metric("f1_score",  f1)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues')
    plt.savefig("confusion_matrix.png", dpi=120); plt.close()
    mlflow.log_artifact("confusion_matrix.png", "plots")

    # Classification Report
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    with open("classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact("classification_report.json", "reports")

    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print("✅ Training selesai!")


if __name__ == "__main__":
    main()
