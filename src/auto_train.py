"""
Auto-train models by fetching sample data from Open Food Facts API.
Used when pre-trained model files are not available (e.g., first deploy on Streamlit Cloud).
"""
import os
import time
import requests
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
USER_AGENT = "SmartFoodScanner/1.0 (auto-train)"

NUMERIC_FEATURES = [
    "energy-kcal_100g", "fat_100g", "saturated-fat_100g", "trans-fat_100g",
    "carbohydrates_100g", "sugars_100g", "fiber_100g", "proteins_100g",
    "salt_100g", "sodium_100g", "additives_n", "nutriscore_score",
    "serving_quantity", "vitamin-a_100g", "vitamin-c_100g",
    "calcium_100g", "iron_100g", "cholesterol_100g"
]

TEXT_FIELDS = ["ingredients_text", "product_name", "labels", "categories"]

FIELDS_PARAM = (
    "product_name,brands,ingredients_text,nutriscore_grade,nova_group,"
    "categories,labels,energy-kcal_100g,fat_100g,saturated-fat_100g,"
    "trans-fat_100g,carbohydrates_100g,sugars_100g,fiber_100g,"
    "proteins_100g,salt_100g,sodium_100g,additives_n,nutriscore_score,"
    "serving_quantity,vitamin-a_100g,vitamin-c_100g,calcium_100g,"
    "iron_100g,cholesterol_100g"
)


def fetch_training_data(target_count=5000, max_pages=60):
    """Fetch products from Open Food Facts API with valid Nutri-Score and NOVA."""
    products = []
    page = 1
    headers = {"User-Agent": USER_AGENT}

    while len(products) < target_count and page <= max_pages:
        url = "https://world.openfoodfacts.org/cgi/search.pl"
        params = {
            "action": "process",
            "json": 1,
            "page_size": 100,
            "page": page,
            "fields": FIELDS_PARAM,
            "tagtype_0": "nutrition_grades",
            "tag_contains_0": "contains",
            "tag_0": "a",
        }
        # Cycle through nutri-score grades to get balanced data
        grade = ["a", "b", "c", "d", "e"][(page - 1) % 5]
        params["tag_0"] = grade

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("products", [])
            for p in batch:
                ns = str(p.get("nutriscore_grade", "")).strip().lower()
                nova = p.get("nova_group")
                ing = p.get("ingredients_text", "")
                if ns in ["a", "b", "c", "d", "e"] and nova in [1, 2, 3, 4] and ing and len(str(ing)) > 10:
                    products.append(p)
        except Exception as e:
            print(f"  Fetch error page {page}: {e}")

        page += 1
        time.sleep(0.3)

        if page % 10 == 0:
            print(f"  Fetched {len(products)} valid products so far...")

    print(f"  Total fetched: {len(products)} products")
    return products


def combine_text(product):
    parts = []
    for field in TEXT_FIELDS:
        val = str(product.get(field, "")).strip()
        if val and val.lower() not in ("nan", "none", ""):
            parts.append(val)
    return " ".join(parts)


def auto_train():
    """Fetch data and train models automatically."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Fetching training data from Open Food Facts API...")
    products = fetch_training_data(target_count=5000)

    if len(products) < 100:
        print("Not enough data fetched. Cannot train models.")
        return False

    # Build DataFrame
    df = pd.DataFrame(products)
    df["nutriscore_grade"] = df["nutriscore_grade"].astype(str).str.strip().str.lower()
    df["nova_group"] = pd.to_numeric(df["nova_group"], errors="coerce").astype(int)
    df["combined_text"] = df.apply(lambda row: combine_text(row), axis=1)

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0.0

    print(f"Training on {len(df)} products...")
    print(f"  Nutri-Score distribution: {df['nutriscore_grade'].value_counts().to_dict()}")
    print(f"  NOVA distribution: {df['nova_group'].value_counts().to_dict()}")

    # Scale numeric
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[NUMERIC_FEATURES].values)

    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2),
        min_df=2, max_df=0.95, sublinear_tf=True
    )
    X_text = tfidf.fit_transform(df["combined_text"])

    # Combine
    X = sp.hstack([X_text, sp.csr_matrix(X_num)])

    # Train Nutri-Score
    print("Training Nutri-Score model...")
    ns_model = LogisticRegression(
        max_iter=1000, class_weight="balanced",
        multi_class="multinomial", solver="lbfgs", random_state=42, C=1.0
    )
    ns_model.fit(X, df["nutriscore_grade"])

    # Train NOVA
    print("Training NOVA Group model...")
    nova_model = LogisticRegression(
        max_iter=1000, class_weight="balanced",
        multi_class="multinomial", solver="lbfgs", random_state=42, C=1.0
    )
    nova_model.fit(X, df["nova_group"])

    # Save
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf.joblib"))
    joblib.dump(ns_model, os.path.join(MODEL_DIR, "ns_model.joblib"))
    joblib.dump(nova_model, os.path.join(MODEL_DIR, "nova_model.joblib"))
    print(f"Models saved to {MODEL_DIR}/")
    return True


if __name__ == "__main__":
    auto_train()
