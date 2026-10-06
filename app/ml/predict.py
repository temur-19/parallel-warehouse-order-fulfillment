from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

MODEL_PATH = Path(__file__).resolve().with_name("model.pkl")


@lru_cache(maxsize=1)
def _load_model() -> Pipeline:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run `python -m app.ml.train` first."
        )
    model: Pipeline = joblib.load(MODEL_PATH)
    return model


def predict_transaction(
    data: Mapping[str, Any], fraud_threshold: float = 0.5
) -> dict[str, float | bool]:
    if not 0.0 <= fraud_threshold <= 1.0:
        raise ValueError("fraud_threshold must be between 0 and 1.")

    model = _load_model()
    feature_columns = list(model.feature_names_in_)
    missing_columns = [column for column in feature_columns if column not in data]
    if missing_columns:
        raise ValueError(f"Missing required transaction features: {missing_columns}")

    transaction = pd.DataFrame(
        [{column: data[column] for column in feature_columns}],
        columns=feature_columns,
    )
    classifier = model.named_steps["classifier"]
    fraud_class_indices = np.flatnonzero(classifier.classes_ == 1)
    if len(fraud_class_indices) != 1:
        raise ValueError("The trained model does not contain fraud class 1.")

    risk_score = float(
        model.predict_proba(transaction)[0, fraud_class_indices[0]]
    )
    return {
        "risk_score": risk_score,
        "is_fraud": risk_score >= fraud_threshold,
    }
