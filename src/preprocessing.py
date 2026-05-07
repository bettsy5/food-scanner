"""
Data preprocessing pipeline for the food scanner.
Handles feature extraction from raw product data.
"""
import numpy as np
import pandas as pd
from typing import Optional


# Structured numeric features used by the model
NUMERIC_FEATURES = [
    "energy-kcal_100g", "fat_100g", "saturated-fat_100g", "trans-fat_100g",
    "carbohydrates_100g", "sugars_100g", "fiber_100g", "proteins_100g",
    "salt_100g", "sodium_100g", "additives_n", "nutriscore_score",
    "serving_quantity", "vitamin-a_100g", "vitamin-c_100g",
    "calcium_100g", "iron_100g", "cholesterol_100g"
]

# Text fields to combine for NLP features
TEXT_FIELDS = ["ingredients_text", "product_name", "labels", "categories"]


def extract_numeric_features(product: dict) -> np.ndarray:
    """Extract structured numeric features from a product dict."""
    values = []
    for feat in NUMERIC_FEATURES:
        val = product.get(feat)
        if val is None or val == "":
            values.append(0.0)
        else:
            try:
                values.append(float(val))
            except (ValueError, TypeError):
                values.append(0.0)
    return np.array(values, dtype=np.float32)


def extract_text(product: dict) -> str:
    """Combine text fields from a product into a single string."""
    parts = []
    for field in TEXT_FIELDS:
        val = product.get(field, "")
        if val and str(val).strip().lower() not in ("nan", "none", ""):
            parts.append(str(val).strip())
    return " ".join(parts)


def extract_features_from_manual_input(
    ingredients: str,
    product_name: str = "",
    energy_kcal: float = 0,
    fat: float = 0,
    saturated_fat: float = 0,
    trans_fat: float = 0,
    carbs: float = 0,
    sugars: float = 0,
    fiber: float = 0,
    protein: float = 0,
    salt: float = 0,
    sodium: float = 0,
) -> tuple[np.ndarray, str]:
    """Build features from manually entered product info."""
    numeric = np.array([
        energy_kcal, fat, saturated_fat, trans_fat,
        carbs, sugars, fiber, protein,
        salt, sodium,
        0,   # additives_n
        0,   # nutriscore_score (unknown)
        0,   # serving_quantity
        0, 0, 0, 0, 0  # vitamins/minerals
    ], dtype=np.float32)

    text = f"{product_name} {ingredients}".strip()
    return numeric, text
