from __future__ import annotations

import warnings
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from parallel_reduce import ParallelReduceClassifier


# ============================================================
# SOZLAMALAR
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "PS_20174392719_1491204439457_log.csv"
)

MODEL_PATH = Path(__file__).resolve().with_name("reduce_model.pkl")

TARGET_COLUMN = "isFraud"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TOTAL_ESTIMATORS = 100

SAMPLE_ROWS_FOR_SCHEMA = 50_000
MAX_CATEGORICAL_CARDINALITY = 100


def _fit_model_chunk(
    model_template,
    X_train,
    y_train,
    chunk_indices,
    n_estimators,
    random_state,
):
    model = clone(model_template)
    model.set_params(
        classifier__n_estimators=n_estimators,
        classifier__random_state=random_state,
    )
    model.fit(
        X_train.iloc[chunk_indices],
        y_train.iloc[chunk_indices],
    )
    return model


# ============================================================
# XUSUSIYATLARNI ANIQLASH
# ============================================================

def get_feature_columns(sample, target_column):
    if target_column not in sample.columns:
        raise ValueError(
            f"Datasetda '{target_column}' ustuni topilmadi!"
        )

    candidate_columns = [
        column
        for column in sample.columns
        if column != target_column
    ]

    categorical_features = []
    numeric_features = []
    excluded_features = []

    for column in candidate_columns:
        dtype = sample[column].dtype

        if (
            pd.api.types.is_object_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(dtype)
        ):
            unique_count = sample[column].nunique(dropna=True)

            # Juda ko'p turli qiymatga ega kategorik ustunlarni chiqarib tashlash
            if unique_count > MAX_CATEGORICAL_CARDINALITY:
                excluded_features.append(column)
                continue

            categorical_features.append(column)

        elif pd.api.types.is_numeric_dtype(dtype):
            numeric_features.append(column)

        else:
            excluded_features.append(column)

    feature_columns = numeric_features + categorical_features

    if not feature_columns:
        raise ValueError("Model uchun mos xususiyatlar topilmadi!")

    print("\n" + "=" * 60)
    print("XUSUSIYATLAR TAHLILI")
    print("=" * 60)

    print(f"Raqamli ustunlar soni: {len(numeric_features)}")
    print(f"Kategorik ustunlar soni: {len(categorical_features)}")
    print(f"Chiqarib tashlangan ustunlar: {excluded_features}")

    print(f"\nRaqamli ustunlar: {numeric_features}")
    print(f"Kategorik ustunlar: {categorical_features}")

    return feature_columns, numeric_features, categorical_features


# ============================================================
# PARALLEL MODEL O'QITISH
# ============================================================

def train_model():
    total_start = perf_counter()

    print("\n" + "=" * 60)
    print("PARALLEL FRAUD DETECTION MODEL TRAINING")
    print("=" * 60)

    # 1. Dataset mavjudligini tekshirish
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset topilmadi: {DATASET_PATH}"
        )

    print(f"\nDataset: {DATASET_PATH}")
    print(f"Model saqlanadigan joy: {MODEL_PATH}")

    # 2. Datasetning kichik qismini o'qish
    print("\nDataset strukturasi aniqlanmoqda...")

    sample = pd.read_csv(
        DATASET_PATH,
        nrows=SAMPLE_ROWS_FOR_SCHEMA,
    )

    original_columns = sample.columns.tolist()

    # 3. Feature ustunlarini aniqlash
    (
        feature_columns,
        numeric_features,
        categorical_features,
    ) = get_feature_columns(sample, TARGET_COLUMN)

    del sample

    # 4. Kerakli ustunlarni to'liq o'qish
    print("\nDataset to'liq yuklanmoqda...")

    data = pd.read_csv(
        DATASET_PATH,
        usecols=feature_columns + [TARGET_COLUMN],
    )

    print("\n" + "=" * 60)
    print("DATASET MA'LUMOTLARI")
    print("=" * 60)

    print(f"Qatorlar soni: {len(data):,}")
    print(f"Datasetdagi original ustunlar: {len(original_columns)}")
    print(f"Model uchun feature ustunlari: {len(feature_columns)}")
    print(f"Target ustuni: {TARGET_COLUMN}")

    # 5. Target qiymatlarini tekshirish
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"'{TARGET_COLUMN}' ustunida bo'sh qiymatlar mavjud!"
        )

    data[TARGET_COLUMN] = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="raise",
    )

    unique_targets = set(data[TARGET_COLUMN].unique())

    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"Target faqat 0 va 1 bo'lishi kerak. "
            f"Topilgan qiymatlar: {unique_targets}"
        )

    if len(unique_targets) != 2:
        raise ValueError(
            "Datasetda ikkala sinf ham bo'lishi kerak: "
            "0 — oddiy tranzaksiya, 1 — firibgarlik."
        )

    # 6. X va y ni ajratish
    X = data[feature_columns]
    y = data[TARGET_COLUMN].astype("int8")

    print("\nTarget taqsimoti:")
    print(y.value_counts().sort_index().rename({
        0: "Oddiy tranzaksiyalar (0)",
        1: "Firibgarlik (1)",
    }))

    del data

    # 7. Train va test qismlariga ajratish
    print("\nTrain va test ma'lumotlari ajratilmoqda...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test:  {y_test.shape}")

    del X, y

    # 8. Raqamli ustunlar uchun preprocessing
    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ])

    # 9. Kategorik ustunlar uchun preprocessing
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=10,
                max_categories=100,
            ),
        ),
    ])

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )

    # 10. Each worker trains a model on a separate stratified data chunk.
    model_template = Pipeline([
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=TOTAL_ESTIMATORS,
                max_depth=16,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    # ========================================================
    # PARALLEL REDUCE O'QITISH
    # ========================================================

    available_workers = joblib.cpu_count()
    minority_class_count = int(y_train.value_counts().min())
    worker_count = min(
        available_workers,
        minority_class_count,
        TOTAL_ESTIMATORS,
    )

    print("\n" + "=" * 60)
    print("PARALLEL REDUCE O'QITISH BOSHLANDI")
    print("=" * 60)

    print(f"Mavjud CPU workerlar: {available_workers}")

    training_start = perf_counter()

    if worker_count < 2:
        chunks = [np.arange(len(y_train))]
    else:
        splitter = StratifiedKFold(
            n_splits=worker_count,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        chunks = [
            chunk_indices
            for _, chunk_indices in splitter.split(X_train, y_train)
        ]

    base_estimators, extra_estimators = divmod(
        TOTAL_ESTIMATORS,
        len(chunks),
    )
    trees_per_chunk = [
        base_estimators + (index < extra_estimators)
        for index in range(len(chunks))
    ]

    print(f"Parallel workerlar: {len(chunks)}")
    print(f"Random Forest daraxtlari jami: {sum(trees_per_chunk)}")
    print(f"Daraxtlar bo'laklar bo'yicha: {trees_per_chunk}")

    fitted_estimators = Parallel(
        n_jobs=len(chunks),
        prefer="threads",
    )(
        delayed(_fit_model_chunk)(
            model_template,
            X_train,
            y_train,
            chunk_indices,
            n_estimators,
            RANDOM_STATE + index,
        )
        for index, (chunk_indices, n_estimators) in enumerate(
            zip(chunks, trees_per_chunk)
        )
    )
    model = ParallelReduceClassifier(fitted_estimators)

    parallel_training_seconds = (
        perf_counter() - training_start
    )

    print("\n" + "=" * 60)
    print("PARALLEL REDUCE O'QITISH TUGADI")
    print("=" * 60)

    print(
        f"O'qitish vaqti: "
        f"{parallel_training_seconds:.2f} sekund"
    )

    print(
        f"O'qitish vaqti: "
        f"{parallel_training_seconds / 60:.2f} minut"
    )

    print(f"Amalda ishlatilgan workerlar: {len(chunks)}")

    # ========================================================
    # MODELNI BAHOLASH
    # ========================================================

    print("\nTest ma'lumotlari orqali model tekshirilmoqda...")

    y_pred = model.predict(X_test)

    classes = list(model.classes_)
    fraud_class_index = classes.index(1)

    y_proba = model.predict_proba(X_test)[:, fraud_class_index]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    auc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 60)
    print("MODEL BAHOLASH NATIJALARI")
    print("=" * 60)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ========================================================
    # MODELNI SAQLASH
    # ========================================================

    print("\nModel saqlanmoqda...")

    joblib.dump(model, MODEL_PATH)

    total_seconds = perf_counter() - total_start

    print("\n" + "=" * 60)
    print("MODEL MUVAFFAQIYATLI SAQLANDI")
    print("=" * 60)

    print(f"Model manzili: {MODEL_PATH}")
    print(f"Fayl mavjud: {MODEL_PATH.exists()}")
    print(f"Umumiy bajarilish vaqti: {total_seconds:.2f} sekund")

    return model


# ============================================================
# DASTURNI ISHGA TUSHIRISH
# ============================================================

if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
    )

    train_model()