from __future__ import annotations

import sys
import warnings
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
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
from threadpoolctl import threadpool_limits

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ml.parallel_reduce import ParallelReduceClassifier


# ============================================================
# SOZLAMALAR
# ============================================================

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
MAX_ITERATIONS = 100

SAMPLE_ROWS_FOR_SCHEMA = 50_000
MAX_CATEGORICAL_CARDINALITY = 100


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




def load_training_data():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset topilmadi: {DATASET_PATH}"
        )

    print("\nDataset strukturasi aniqlanmoqda...")
    sample = pd.read_csv(
        DATASET_PATH,
        nrows=SAMPLE_ROWS_FOR_SCHEMA,
    )
    original_columns = sample.columns.tolist()
    (
        feature_columns,
        numeric_features,
        categorical_features,
    ) = get_feature_columns(sample, TARGET_COLUMN)
    del sample

    print("\nDataset to'liq yuklanmoqda...")
    data = pd.read_csv(
        DATASET_PATH,
        usecols=feature_columns + [TARGET_COLUMN],
    )
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

    X = data[feature_columns]
    y = data[TARGET_COLUMN].astype("int8")
    print("\nTarget taqsimoti:")
    print(y.value_counts().sort_index().rename({
        0: "Oddiy tranzaksiyalar (0)",
        1: "Firibgarlik (1)",
    }))
    del data

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
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_columns,
        numeric_features,
        categorical_features,
        original_columns,
    )


def create_model_template(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ])

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
                sparse_output=False,
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
        sparse_threshold=0.0,
    )

    return Pipeline([
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            HistGradientBoostingClassifier(
                max_iter=MAX_ITERATIONS,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def report_and_save_model(model, X_test, y_test, total_start):
    print("\nTest ma'lumotlari orqali model tekshirilmoqda...")
    y_pred = model.predict(X_test)
    fraud_class_index = list(model.classes_).index(1)
    y_proba = model.predict_proba(X_test)[:, fraud_class_index]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
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




def train_mpi_model(total_start):
    try:
        from mpi4py import MPI
    except ImportError as error:
        raise RuntimeError(
            "MPI training requires mpi4py and an MPI runtime. "
            "Install project dependencies and an MPI implementation, then "
            "launch with `mpiexec -n 4 python -m app.ml.train`."
        ) from error

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    setup_error = None
    root_exception = None
    payloads = None
    model_template = None
    X_test = None
    y_test = None

    if rank == 0:
        try:
            (
                X_train,
                X_test,
                y_train,
                y_test,
                _,
                numeric_features,
                categorical_features,
                _,
            ) = load_training_data()
            if int(y_train.value_counts().min()) < world_size:
                raise ValueError(
                    "MPI processlar soni har bir sinfdagi train "
                    "namunalari sonidan oshmasligi kerak."
                )
            if world_size == 1:
                payloads = [(X_train, y_train)]
            else:
                splitter = StratifiedKFold(
                    n_splits=world_size,
                    shuffle=True,
                    random_state=RANDOM_STATE,
                )
                payloads = [
                    (
                        X_train.iloc[indices].copy(),
                        y_train.iloc[indices].copy(),
                    )
                    for _, indices in splitter.split(X_train, y_train)
                ]
            model_template = create_model_template(
                numeric_features,
                categorical_features,
            )
        except Exception as error:
            root_exception = error
            setup_error = f"{type(error).__name__}: {error}"

    setup_error = comm.bcast(setup_error, root=0)
    if setup_error is not None:
        if root_exception is not None:
            raise root_exception
        raise RuntimeError(f"MPI training setup failed on rank 0: {setup_error}")

    model_template = comm.bcast(model_template, root=0)
    local_X, local_y = comm.scatter(payloads, root=0)

    print(
        f"MPI rank {rank}/{world_size}: "
        f"{len(local_y):,} train qatori, OpenMP=1 thread"
    )
    training_start = perf_counter()
    with threadpool_limits(limits=1, user_api="openmp"):
        model_template.fit(local_X, local_y)
    local_training_seconds = perf_counter() - training_start
    print(
        f"MPI rank {rank}: o'qitish "
        f"{local_training_seconds:.2f} sekund"
    )

    fitted_models = comm.gather(model_template, root=0)
    if rank != 0:
        return None

    model = ParallelReduceClassifier(fitted_models)
    print(
        f"MPI orqali {world_size} model birlashtirildi. "
        "Har bir model alohida stratified data shardida o'qitilgan."
    )
    return report_and_save_model(model, X_test, y_test, total_start)


def train_model():
    total_start = perf_counter()
    print("\n" + "=" * 60)
    print("MPI PARALLEL FRAUD DETECTION MODEL TRAINING")
    print("=" * 60)
    print(f"\nDataset: {DATASET_PATH}")
    print(f"Model saqlanadigan joy: {MODEL_PATH}")
    return train_mpi_model(total_start)




if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
    )
    train_model()

