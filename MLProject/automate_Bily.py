"""
automate_Bily.py
Otomatisasi preprocessing dataset Breast Cancer Wisconsin.
Jalankan: python automate_Bily.py
"""

import pandas as pd
import numpy as np
import os
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

RAW_DIR    = "../breast_cancer_raw"
OUTPUT_DIR = "../breast_cancer_preprocessing"
TEST_SIZE  = 0.2
RANDOM_STATE = 42


def load_dataset():
    print("[1/5] Memuat dataset...")
    bc = load_breast_cancer()
    df = pd.DataFrame(data=bc.data, columns=bc.feature_names)
    df['target']    = bc.target
    df['diagnosis'] = df['target'].map({0: 'malignant', 1: 'benign'})
    os.makedirs(RAW_DIR, exist_ok=True)
    df.to_csv(f"{RAW_DIR}/breast_cancer_raw.csv", index=False)
    print(f"    Dataset dimuat: {df.shape} → disimpan ke {RAW_DIR}/breast_cancer_raw.csv")
    return df, list(bc.feature_names)


def remove_duplicates(df):
    print("[2/5] Menghapus duplikat...")
    before = len(df)
    df = df.drop_duplicates()
    print(f"    Duplikat dihapus: {before - len(df)} baris")
    return df


def handle_missing_values(df, feature_cols):
    print("[3/5] Menangani missing values...")
    for col in feature_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    print(f"    Total missing values: {df[feature_cols].isnull().sum().sum()}")
    return df


def remove_outliers_iqr(df, feature_cols):
    print("[4/5] Menangani outlier (IQR)...")
    total = 0
    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        removed = before - len(df)
        total += removed
    print(f"    Total outlier dihapus: {total} | Shape: {df.shape}")
    return df


def normalize_and_split(df, feature_cols):
    print("[5/5] Normalisasi & split data...")
    X = df[feature_cols].copy()
    y = df['target'].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )
    print(f"    Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def save_preprocessed(X_train, X_test, y_train, y_test):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df = X_train.copy()
    train_df['target'] = y_train.values
    test_df = X_test.copy()
    test_df['target'] = y_test.values
    train_df.to_csv(f"{OUTPUT_DIR}/breast_cancer_train.csv", index=False)
    test_df.to_csv(f"{OUTPUT_DIR}/breast_cancer_test.csv", index=False)
    print(f"\n✅ Preprocessing selesai!")
    print(f"   breast_cancer_train.csv → {train_df.shape}")
    print(f"   breast_cancer_test.csv  → {test_df.shape}")


def run_preprocessing():
    print("=" * 55)
    print("  PIPELINE PREPROCESSING - BREAST CANCER DATASET")
    print("=" * 55)
    df, feature_cols = load_dataset()
    df = remove_duplicates(df)
    df = handle_missing_values(df, feature_cols)
    df = remove_outliers_iqr(df, feature_cols)
    X_train, X_test, y_train, y_test = normalize_and_split(df, feature_cols)
    save_preprocessed(X_train, X_test, y_train, y_test)
    print("=" * 55)


if __name__ == "__main__":
    run_preprocessing()
