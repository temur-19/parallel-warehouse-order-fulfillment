from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, TypedDict

import joblib
import numpy as np
import torch
from torch import nn

MODEL_PATH = Path(__file__).resolve().with_name("cuda_train_fast.pkl")


class PredictionReason(TypedDict):
    feature: str | None
    value: float | str | None
    baseline: float | str | None
    impact_percentage_points: float
    description: str


class PredictionResult(TypedDict):
    risk_score: float
    model_risk_score: float
    contextual_risk_score: float
    fraud_threshold: float
    review_threshold: float
    is_fraud: bool
    reasons: list[PredictionReason]


REVIEW_RISK_THRESHOLD = 0.50


class _CudaFraudModel(nn.Module):
    def __init__(
        self,
        numeric_feature_count: int,
        category_count: int,
        hidden_layers: tuple[int, ...],
        embedding_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.category_embedding = nn.Embedding(
            category_count + 1, embedding_dimension
        )
        layers: list[nn.Module] = []
        input_size = numeric_feature_count + embedding_dimension
        for hidden_size in hidden_layers:
            layers.extend(
                (
                    nn.Linear(input_size, hidden_size, bias=False),
                    nn.BatchNorm1d(hidden_size),
                    nn.SiLU(inplace=True),
                    nn.Dropout(dropout),
                )
            )
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(
        self, numeric_features: torch.Tensor, category_ids: torch.Tensor
    ) -> torch.Tensor:
        embedded = self.category_embedding(category_ids)
        combined = torch.cat((numeric_features, embedded), dim=1)
        return self.layers(combined).squeeze(1)


@dataclass(frozen=True)
class _LoadedModel:
    model: _CudaFraudModel
    numeric_features: tuple[str, ...]
    category_feature: str
    category_ids: dict[str, int]
    means: torch.Tensor
    standard_deviations: torch.Tensor
    fraud_threshold: float


FEATURE_LABELS = {
    "type": "Tranzaksiya turi",
    "step": "Tranzaksiya vaqti (step)",
    "amount": "Tranzaksiya summasi",
    "oldbalanceOrg": "Jo'natuvchining avvalgi balansi",
    "oldbalanceDest": "Qabul qiluvchining avvalgi balansi",
    "isFlaggedFraud": "Manbadagi firibgarlik belgisi",
}

CONTEXT_FEATURE_LABELS = {
    "is_new_device": "Yangi qurilma",
    "transactions_last_10_min": "Oxirgi 10 daqiqadagi urinishlar",
}


@lru_cache(maxsize=1)
def _load_model() -> _LoadedModel:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained CUDA model not found at {MODEL_PATH}.")

    artifact: Any = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict) or artifact.get("format_version") != 3:
        raise ValueError(
            f"Unsupported CUDA model artifact format in {MODEL_PATH}."
        )

    config = artifact["model_config"]
    numeric_features = tuple(config["raw_numeric_features"])
    categories = tuple(config["categories"])
    hidden_layers = tuple(config["hidden_layers"])
    embedding_dimension = int(config["embedding_dimension"])
    dropout = float(config["dropout"])
    category_feature = str(config["categorical_feature"])

    normalization = artifact["normalization"]
    means = torch.as_tensor(normalization["means"], dtype=torch.float32)
    standard_deviations = torch.as_tensor(
        normalization["standard_deviations"], dtype=torch.float32
    )
    if (
        not numeric_features
        or not hidden_layers
        or len(numeric_features) != means.numel()
        or means.shape != standard_deviations.shape
        or not bool(torch.isfinite(means).all())
        or not bool(torch.isfinite(standard_deviations).all())
        or bool(torch.any(standard_deviations <= 0))
    ):
        raise ValueError("CUDA model normalization does not match its features.")

    model = _CudaFraudModel(
        numeric_feature_count=len(numeric_features),
        category_count=len(categories),
        hidden_layers=hidden_layers,
        embedding_dimension=embedding_dimension,
        dropout=dropout,
    )
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()

    fraud_threshold = float(artifact["threshold"])
    if not np.isfinite(fraud_threshold) or not 0.0 <= fraud_threshold <= 1.0:
        raise ValueError("CUDA model fraud threshold must be between 0 and 1.")

    return _LoadedModel(
        model=model,
        numeric_features=numeric_features,
        category_feature=category_feature,
        category_ids={
            category: index + 1 for index, category in enumerate(categories)
        },
        means=means,
        standard_deviations=standard_deviations,
        fraud_threshold=fraud_threshold,
    )


async def predict_transaction(
    data: Mapping[str, Any], fraud_threshold: float | None = None
) -> PredictionResult:
    trained = _load_model()
    threshold = (
        trained.fraud_threshold if fraud_threshold is None else fraud_threshold
    )
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("fraud_threshold must be between 0 and 1.")

    required_features = (*trained.numeric_features, trained.category_feature)
    missing_features = [name for name in required_features if name not in data]
    if missing_features:
        raise ValueError(
            f"Missing required transaction features: {missing_features}"
        )

    try:
        numeric_values = [float(data[name]) for name in trained.numeric_features]
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Numeric transaction features must be valid numbers."
        ) from error
    if not np.isfinite(numeric_values).all():
        raise ValueError("Numeric transaction features must be finite.")

    category_value = data[trained.category_feature]
    if not isinstance(category_value, str):
        raise ValueError(
            f"Categorical transaction feature '{trained.category_feature}' "
            "must be a string."
        )

    recent_transaction_count = data.get("transactions_last_10_min", 0)
    is_new_device = data.get("is_new_device", False)
    if (
        isinstance(recent_transaction_count, bool)
        or not isinstance(recent_transaction_count, int)
        or recent_transaction_count < 0
    ):
        raise ValueError("transactions_last_10_min must be a non-negative integer.")
    if not isinstance(is_new_device, bool):
        raise ValueError("is_new_device must be a boolean.")

    numeric_tensor = torch.tensor(numeric_values, dtype=torch.float32)
    normalized_features = (
        numeric_tensor - trained.means
    ) / trained.standard_deviations
    category_id = trained.category_ids.get(category_value, 0)
    category_tensor = torch.tensor([category_id], dtype=torch.long)

    with torch.inference_mode():
        logits = trained.model(normalized_features.unsqueeze(0), category_tensor)
        model_risk_score = float(torch.sigmoid(logits)[0].item())

        feature_count = len(trained.numeric_features)
        counterfactual_features = normalized_features.repeat(feature_count + 1, 1)
        for index in range(feature_count):
            counterfactual_features[index, index] = 0.0
        counterfactual_categories = torch.cat(
            (
                category_tensor.expand(feature_count),
                torch.zeros(1, dtype=torch.long),
            )
        )
        counterfactual_scores = torch.sigmoid(
            trained.model(
                counterfactual_features, counterfactual_categories
            )
        )

    impacts = [
        (
            FEATURE_LABELS.get(feature, feature),
            numeric_values[index],
            float(trained.means[index].item()),
            model_risk_score - float(counterfactual_scores[index].item()),
        )
        for index, feature in enumerate(trained.numeric_features)
    ]
    category_impact = model_risk_score - float(counterfactual_scores[-1].item())
    impacts.append(
        (
            trained.category_feature,
            category_value,
            None,
            category_impact,
        )
    )
    reasons: list[PredictionReason] = []
    for label, value, average, impact in sorted(
        impacts, key=lambda item: item[3], reverse=True
    ):
        if impact < 0.005 or len(reasons) == 3:
            continue
        is_numeric = average is not None
        reasons.append(
            {
                "feature": label,
                "value": value,
                "baseline": average,
                "impact_percentage_points": round(impact * 100, 1),
                "description": (
                    f"O'quv ma'lumotidagi o'rtacha qiymat: {average:,.2f}"
                    if is_numeric
                    else "Noma'lum tranzaksiya turiga nisbatan taqqoslandi."
                ),
            }
        )
    if not reasons:
        reasons.append(
            {
                "feature": None,
                "value": None,
                "baseline": None,
                "impact_percentage_points": 0.0,
                "description": (
                    "Alohida bitta xususiyat risk bahosini sezilarli oshirmadi; "
                    "model barcha xususiyatlar uyg'unligiga qarab baho berdi."
                ),
            }
        )

    contextual_risk = min(
        0.70,
        (0.20 if is_new_device else 0.0)
        + min(recent_transaction_count, 5) * 0.10,
    )
    risk_score = max(model_risk_score, contextual_risk)
    contextual_impact_ratio = (
        max(0.0, contextual_risk - model_risk_score) / contextual_risk
        if contextual_risk
        else 0.0
    )
    if is_new_device:
        reasons.append(
            {
                "feature": CONTEXT_FEATURE_LABELS["is_new_device"],
                "value": "Ha",
                "baseline": None,
                "impact_percentage_points": round(
                    0.20 * contextual_impact_ratio * 100, 1
                ),
                "description": (
                    "Qurilma ushbu foydalanuvchi uchun bazada avval "
                    "ro‘yxatdan o‘tmagan."
                ),
            }
        )
    if recent_transaction_count:
        reasons.append(
            {
                "feature": CONTEXT_FEATURE_LABELS["transactions_last_10_min"],
                "value": recent_transaction_count,
                "baseline": 0,
                "impact_percentage_points": round(
                    min(recent_transaction_count, 5)
                    * 0.10
                    * contextual_impact_ratio
                    * 100,
                    1,
                ),
                "description": (
                    "Qiymat ushbu foydalanuvchining oxirgi 10 daqiqadagi "
                    "urinishlari sonidan olindi."
                ),
            }
        )

    return {
        "risk_score": risk_score,
        "model_risk_score": model_risk_score,
        "contextual_risk_score": contextual_risk,
        "fraud_threshold": threshold,
        "review_threshold": REVIEW_RISK_THRESHOLD,
        "is_fraud": model_risk_score >= threshold,
        "reasons": reasons,
    }
