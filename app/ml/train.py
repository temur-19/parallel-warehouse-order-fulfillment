from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "PS_20174392719_1491204439457_log.csv"
)
MODEL_PATH = Path(__file__).resolve().with_name("model.pkl")
TARGET_COLUMN = "isFraud"
RANDOM_STATE = 42
TEST_SIZE = 0.2
SAMPLE_ROWS_FOR_SCHEMA = 50_000
MAX_CATEGORICAL_CARDINALITY = 100


def _get_feature_columns(
    sample: pd.DataFrame,
    target_column: str,
) -> tuple[list[str], list[str], list[str]]:
    if target_column not in sample.columns:
        raise ValueError(
            f"Target column {target_column!r} was not found in the dataset. "
            f"Available columns: {list(sample.columns)}"
        )

    candidate_columns = [column for column in sample.columns if column != target_column]
    categorical_columns = sample[candidate_columns].select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    identifier_columns = [
        column
        for column in categorical_columns
        if sample[column].nunique(dropna=True) > MAX_CATEGORICAL_CARDINALITY
    ]
    feature_columns = [
        column for column in candidate_columns if column not in identifier_columns
    ]
    categorical_features = [
        column for column in categorical_columns if column in feature_columns
    ]
    numeric_features = [
        column for column in feature_columns if column not in categorical_features
    ]

    if not feature_columns:
        raise ValueError(
            "No usable feature columns remain after excluding high-cardinality "
            "categorical identifiers."
        )

    return feature_columns, numeric_features, categorical_features


def train_model(
    dataset_path: Path = DATASET_PATH,
    target_column: str = TARGET_COLUMN,
    model_path: Path = MODEL_PATH,
) -> Pipeline:
    if not dataset_path.is_file():
        available_datasets = sorted(
            path.name for path in dataset_path.parent.glob("*.csv")
        )
        available_text = (
            ", ".join(available_datasets) if available_datasets else "none"
        )
        raise FileNotFoundError(
            f"Training dataset not found: {dataset_path}\n"
            f"CSV files currently in {dataset_path.parent}: {available_text}\n"
            "Update DATASET_PATH to the CSV location and TARGET_COLUMN to its "
            "label column."
        )

    sample = pd.read_csv(dataset_path, nrows=SAMPLE_ROWS_FOR_SCHEMA)
    feature_columns, numeric_features, categorical_features = _get_feature_columns(
        sample, target_column
    )
    columns_to_load = [*feature_columns, target_column]
    data = pd.read_csv(dataset_path, usecols=columns_to_load)

    if data[target_column].isna().any():
        raise ValueError(f"Target column {target_column!r} contains missing values.")

    X = data[feature_columns]
    y = pd.to_numeric(data[target_column], errors="raise")
    if set(y.unique()) != {0, 1}:
        raise ValueError(
            f"Target column {target_column!r} must use 0 for legitimate and 1 "
            f"for fraud; found: {sorted(y.unique().tolist())}"
        )

    stratify = y if y.value_counts().min() >= 2 else None
    if stratify is None:
        warnings.warn(
            "The least-populated target class has fewer than two rows; "
            "performing a non-stratified train/test split.",
            stacklevel=2,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    max_categories=100,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=16,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    classifier = model.named_steps["classifier"]
    fraud_class_indices = np.flatnonzero(classifier.classes_ == 1)
    fraud_probabilities = model.predict_proba(X_test)[:, fraud_class_indices[0]]

    print(f"Dataset shape: ({len(data)}, {len(sample.columns)})")
    print(f"Features: {feature_columns}")
    print(f"Target: {target_column}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Class distribution: {y.value_counts().sort_index().to_dict()}")
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    print()
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score: {f1_score(y_test, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, fraud_probabilities):.4f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    try:
        displayed_model_path = model_path.relative_to(PROJECT_ROOT)
    except ValueError:
        displayed_model_path = model_path
    print(f"\nModel saved to: {displayed_model_path}")
    return model


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
