"""
Train and save TF-IDF + Logistic Regression models for Nutri-Score and NOVA prediction.

Usage:
    python train_models.py --off-path path/to/en.openfoodfacts.org.products.csv

This script:
  1. Loads Open Food Facts data (selecting only needed columns)
  2. Filters to valid Nutri-Score (a-e) and NOVA (1-4) labels
  3. Fits a StandardScaler on numeric features
  4. Fits a TF-IDF vectorizer on ingredient/product text
  5. Trains Logistic Regression classifiers for both targets
  6. Saves all artifacts to models/
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp


NUMERIC_FEATURES = [
    "energy-kcal_100g", "fat_100g", "saturated-fat_100g", "trans-fat_100g",
    "carbohydrates_100g", "sugars_100g", "fiber_100g", "proteins_100g",
    "salt_100g", "sodium_100g", "additives_n", "nutriscore_score",
    "serving_quantity", "vitamin-a_100g", "vitamin-c_100g",
    "calcium_100g", "iron_100g", "cholesterol_100g"
]

TEXT_FIELDS = ["ingredients_text", "product_name", "labels_en", "categories_en"]

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def combine_text(row):
    parts = []
    for col in TEXT_FIELDS:
        val = str(row.get(col, "")).strip()
        if val and val.lower() not in ("nan", "none", ""):
            parts.append(val)
    return " ".join(parts)


def main(off_path: str, sample: int = 0):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- Load ---
    cols_needed = NUMERIC_FEATURES + TEXT_FIELDS + ["nutriscore_grade", "nova_group"]
    print(f"Loading {off_path} ...")
    df = pd.read_csv(off_path, sep="\t", usecols=cols_needed,
                     low_memory=False, on_bad_lines="skip")
    print(f"  Loaded {len(df):,} rows")

    # --- Filter valid targets ---
    df["nutriscore_grade"] = df["nutriscore_grade"].astype(str).str.strip().str.lower()
    df = df[df["nutriscore_grade"].isin(["a", "b", "c", "d", "e"])]
    df["nova_group"] = pd.to_numeric(df["nova_group"], errors="coerce")
    df = df[df["nova_group"].isin([1, 2, 3, 4])]
    df["nova_group"] = df["nova_group"].astype(int)
    print(f"  After filtering: {len(df):,} rows with valid targets")

    # --- Text ---
    df["combined_text"] = df.apply(combine_text, axis=1)
    has_text = df["combined_text"].str.len() > 10
    df = df[has_text]
    print(f"  With text: {len(df):,} rows")

    # --- Optional sample ---
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42)
        print(f"  Sampled down to: {len(df):,} rows")

    # --- Numeric ---
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- Train / test split ---
    X_train_df, X_test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["nutriscore_grade"]
    )
    print(f"  Train: {len(X_train_df):,}  Test: {len(X_test_df):,}")

    # --- Scaler ---
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_df[NUMERIC_FEATURES].values)
    X_test_num = scaler.transform(X_test_df[NUMERIC_FEATURES].values)

    # --- TF-IDF ---
    tfidf = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2),
        min_df=5, max_df=0.95, sublinear_tf=True
    )
    X_train_text = tfidf.fit_transform(X_train_df["combined_text"])
    X_test_text = tfidf.transform(X_test_df["combined_text"])

    # --- Combine ---
    X_train = sp.hstack([X_train_text, sp.csr_matrix(X_train_num)])
    X_test = sp.hstack([X_test_text, sp.csr_matrix(X_test_num)])
    print(f"  Combined feature matrix: {X_train.shape}")

    # --- Train Nutri-Score model ---
    print("\nTraining Nutri-Score model...")
    ns_model = LogisticRegression(
        max_iter=1000, class_weight="balanced",
        multi_class="multinomial", solver="lbfgs", random_state=42, C=1.0
    )
    ns_model.fit(X_train, X_train_df["nutriscore_grade"])
    ns_pred = ns_model.predict(X_test)
    print(classification_report(X_test_df["nutriscore_grade"], ns_pred, digits=3))

    # --- Train NOVA model ---
    print("Training NOVA Group model...")
    nova_model = LogisticRegression(
        max_iter=1000, class_weight="balanced",
        multi_class="multinomial", solver="lbfgs", random_state=42, C=1.0
    )
    nova_model.fit(X_train, X_train_df["nova_group"])
    nova_pred = nova_model.predict(X_test)
    print(classification_report(X_test_df["nova_group"], nova_pred, digits=3))

    # --- Save ---
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf.joblib"))
    joblib.dump(ns_model, os.path.join(MODEL_DIR, "ns_model.joblib"))
    joblib.dump(nova_model, os.path.join(MODEL_DIR, "nova_model.joblib"))
    print(f"\nAll models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train food scanner models")
    parser.add_argument("--off-path", required=True,
                        help="Path to en.openfoodfacts.org.products.csv")
    parser.add_argument("--sample", type=int, default=0,
                        help="Sample N rows for faster training (0 = use all)")
    args = parser.parse_args()
    main(args.off_path, args.sample)
