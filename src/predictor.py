"""
Model prediction pipeline.
Loads trained TF-IDF + classifier models and generates predictions.
"""
import os
import joblib
import numpy as np
from typing import Optional

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class FoodPredictor:
    """Predicts Nutri-Score and NOVA Group from product features."""

    def __init__(self):
        self.scaler = None
        self.tfidf = None
        self.ns_model = None
        self.nova_model = None
        self._loaded = False

    def load(self) -> bool:
        """Load trained models from disk."""
        try:
            self.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
            self.tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf.joblib"))
            self.ns_model = joblib.load(os.path.join(MODEL_DIR, "ns_model.joblib"))
            self.nova_model = joblib.load(os.path.join(MODEL_DIR, "nova_model.joblib"))
            self._loaded = True
            return True
        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
            print("Run train_models.py first to generate model files.")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, numeric_features: np.ndarray, text: str) -> dict:
        """
        Predict Nutri-Score and NOVA Group for a single product.

        Args:
            numeric_features: Array of structured numeric features
            text: Combined ingredient/product text

        Returns:
            Dict with predictions and confidence scores
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

        # Scale numeric features
        numeric_scaled = self.scaler.transform(numeric_features.reshape(1, -1))

        # TF-IDF text features
        text_features = self.tfidf.transform([text])

        # Combine: sparse TF-IDF + dense numeric
        import scipy.sparse as sp
        combined = sp.hstack([
            text_features,
            sp.csr_matrix(numeric_scaled)
        ])

        # Nutri-Score prediction
        ns_pred = self.ns_model.predict(combined)[0]
        ns_proba = self.ns_model.predict_proba(combined)[0]
        ns_classes = self.ns_model.classes_

        # NOVA prediction
        nova_pred = self.nova_model.predict(combined)[0]
        nova_proba = self.nova_model.predict_proba(combined)[0]
        nova_classes = self.nova_model.classes_

        return {
            "nutriscore": {
                "grade": str(ns_pred).upper(),
                "confidence": float(np.max(ns_proba)),
                "probabilities": {
                    str(c).upper(): float(p)
                    for c, p in zip(ns_classes, ns_proba)
                }
            },
            "nova": {
                "group": int(nova_pred),
                "confidence": float(np.max(nova_proba)),
                "probabilities": {
                    int(c): float(p)
                    for c, p in zip(nova_classes, nova_proba)
                }
            }
        }
