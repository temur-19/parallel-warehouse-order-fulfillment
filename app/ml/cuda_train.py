from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "PS_20174392719_1491204439457_log.csv"
RESULT_PATH = Path(__file__).resolve().with_name("cuda_train_fast.pkl")

TARGET_COLUMN = "isFraud"
CATEGORICAL_COLUMN = "type"
RAW_NUMERIC_COLUMNS = (
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
)
FEATURE_NAMES = (
    "step",
    "hour_of_day",
    "log_amount",
    "log_oldbalanceOrg",
    "log_newbalanceOrig",
    "log_oldbalanceDest",
    "log_newbalanceDest",
    "log_error_orig",
    "log_error_dest",
    "isFlaggedFraud",
    "orig_balance_zero_after",
    "dest_balance_zero_both",
    "orig_fully_drained",
)

RANDOM_STATE = 42
TEST_SIZE = 0.2  # original skript bilan bir xil test to'plam
VAL_SIZE = 0.1  # train ichidan: threshold tanlash va eng yaxshi epoch uchun
EPOCHS = 20
BATCH_SIZE = 32_768
EVAL_BATCH_SIZE = 262_144
MAX_LEARNING_RATE = 4e-3
WEIGHT_DECAY = 1e-4
EMBEDDING_DIM = 8
HIDDEN_LAYERS = (256, 128, 64)
DROPOUT = 0.1



class FraudMLP(nn.Module):
    def __init__(
        self,
        feature_count: int,
        category_count: int,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_layers: tuple[int, ...] = HIDDEN_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.category_embedding = nn.Embedding(category_count + 1, embedding_dim)
        layers: list[nn.Module] = []
        in_dim = feature_count + embedding_dim
        for hidden in hidden_layers:
            layers += [
                nn.Linear(in_dim, hidden, bias=False),
                nn.BatchNorm1d(hidden),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(
        self, features: torch.Tensor, category_ids: torch.Tensor
    ) -> torch.Tensor:
        embedded = self.category_embedding(category_ids)
        return self.layers(torch.cat((features, embedded), dim=1)).squeeze(1)


# --------------------------------------------------------------------------
# GPU feature engineering (butun dataset bo'yicha parallel, elementwise)
# --------------------------------------------------------------------------
def _signed_log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def build_features(raw: torch.Tensor) -> torch.Tensor:
    """raw: (N, 7) float64 GPU tensor, ustunlar tartibi RAW_NUMERIC_COLUMNS.

    Xato (error) feature'lari float64 da hisoblanadi: float32 da katta
    summalarda markaziy qiymatlar yo'qoladi. Natija float32 ga o'tkaziladi.
    """
    step, amount, old_org, new_org, old_dest, new_dest, flagged = raw.unbind(dim=1)
    error_orig = old_org - amount - new_org
    error_dest = old_dest + amount - new_dest
    columns = (
        step,
        step % 24,
        _signed_log1p(amount),
        _signed_log1p(old_org),
        _signed_log1p(new_org),
        _signed_log1p(old_dest),
        _signed_log1p(new_dest),
        _signed_log1p(error_orig),
        _signed_log1p(error_dest),
        flagged,
        (new_org == 0).double(),
        ((old_dest == 0) & (new_dest == 0)).double(),
        (amount == old_org).double(),
    )
    return torch.stack([column.float() for column in columns], dim=1)



def load_data_to_gpu(device: torch.device) -> dict[str, object]:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset topilmadi: {DATASET_PATH}")

    columns = [*RAW_NUMERIC_COLUMNS, CATEGORICAL_COLUMN, TARGET_COLUMN]
    data = pd.read_csv(DATASET_PATH, usecols=columns)
    target = pd.to_numeric(data[TARGET_COLUMN], errors="raise").to_numpy(
        dtype=np.int8
    )
    if not np.isin(target, (0, 1)).all() or np.unique(target).size != 2:
        raise ValueError(f"'{TARGET_COLUMN}' ustunida 0 va 1 sinflari bo'lishi kerak.")

    all_indices = np.arange(len(target))
    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=target[train_val_indices],
    )

    category_values = data[CATEGORICAL_COLUMN]
    known_categories = sorted(
        category_values.iloc[train_indices].dropna().unique().tolist()
    )
    category_mapping = {name: i + 1 for i, name in enumerate(known_categories)}
    category_ids = (
        category_values.map(category_mapping).fillna(0).to_numpy(dtype=np.int64)
    )
    raw = data.loc[:, RAW_NUMERIC_COLUMNS].to_numpy(dtype=np.float64)
    del data

    # Bir marta H2D ko'chirish; keyin hamma narsa GPU'da.
    raw_gpu = torch.from_numpy(raw).to(device)
    del raw
    features = build_features(raw_gpu)
    del raw_gpu
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Feature'larda NaN yoki cheksiz qiymat bor.")

    category_gpu = torch.from_numpy(category_ids).to(device)
    target_gpu = torch.from_numpy(target.astype(np.float32)).to(device)

    splits: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for name, indices in (
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ):
        index_gpu = torch.from_numpy(indices).to(device)
        splits[name] = (
            features[index_gpu],
            category_gpu[index_gpu],
            target_gpu[index_gpu],
        )
    del features, category_gpu, target_gpu

    means = splits["train"][0].mean(dim=0)
    stds = splits["train"][0].std(dim=0)
    stds = torch.where(stds == 0, torch.ones_like(stds), stds)
    for features_split, _, _ in splits.values():
        features_split.sub_(means).div_(stds)

    return {
        "splits": splits,
        "means": means.cpu().numpy(),
        "stds": stds.cpu().numpy(),
        "categories": known_categories,
    }



def _select_amp_dtype(device: torch.device) -> torch.dtype:
    if torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return torch.float16


def train_one_epoch(
    model: FraudMLP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    loss_function: nn.Module,
    split: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    amp_dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    features, category_ids, target = split
    device = features.device
    model.train()
    sample_count = features.shape[0]
    steps = sample_count // BATCH_SIZE  # BatchNorm uchun to'liq batch'lar
    permutation = torch.randperm(sample_count, device=device, generator=generator)
    total_loss = torch.zeros((), device=device)

    for step in range(steps):
        batch = permutation[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(features[batch], category_ids[batch])
        loss = loss_function(logits.float(), target[batch])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.detach()  # CPU bilan sinxronlashtirmaymiz
    return total_loss / steps


@torch.inference_mode()
def predict_probabilities(
    model: FraudMLP,
    split: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    features, category_ids, _ = split
    model.eval()
    probabilities = torch.empty(features.shape[0], device=features.device)
    for start in range(0, features.shape[0], EVAL_BATCH_SIZE):
        end = start + EVAL_BATCH_SIZE
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(features[start:end], category_ids[start:end])
        probabilities[start:end] = torch.sigmoid(logits.float())
    return probabilities


def find_best_threshold(actual: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(actual, probabilities)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def compute_metrics(
    actual: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | list[list[int]]]:
    predicted = (probabilities >= threshold).astype(np.int8)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1_score": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probabilities)),
        "pr_auc": float(average_precision_score(actual, probabilities)),
        "confusion_matrix": confusion_matrix(actual, predicted).tolist(),
    }


def _cuda_event_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


def train_model() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA mavjud emas. CUDA-enabled PyTorch va mos NVIDIA drayverini "
            "tekshiring; bu skript CPU rejimiga o'tmaydi."
        )

    device = torch.device("cuda")
    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed_all(RANDOM_STATE)
    torch.set_float32_matmul_precision("high")  # TF32 tensor core'lar
    torch.backends.cudnn.benchmark = True
    amp_dtype = _select_amp_dtype(device)
    print(f"Dataset: {DATASET_PATH}")
    print(f"CUDA qurilmasi: {torch.cuda.get_device_name(device)}")
    print(f"AMP dtype: {amp_dtype}")

    from time import perf_counter

    total_start = perf_counter()
    data_start = perf_counter()
    prepared = load_data_to_gpu(device)
    torch.cuda.synchronize(device)
    data_seconds = perf_counter() - data_start

    splits = prepared["splits"]
    known_categories: list[str] = prepared["categories"]  # type: ignore[assignment]
    train_split, val_split, test_split = splits["train"], splits["val"], splits["test"]  # type: ignore[index]
    print(
        f"Train: {train_split[0].shape[0]:,}; val: {val_split[0].shape[0]:,}; "
        f"test: {test_split[0].shape[0]:,}; "
        f"yuklash + GPU feature engineering: {data_seconds:.2f} s"
    )

    model = FraudMLP(len(FEATURE_NAMES), len(known_categories)).to(device)
    positive_count = float(train_split[2].sum().item())
    negative_count = train_split[2].shape[0] - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Train qismida ikkala target sinfi ham bo'lishi kerak.")
    # To'liq nisbat o'rniga ildiz: barqarorroq; qolganini threshold tuning hal qiladi.
    pos_weight = torch.tensor((negative_count / positive_count) ** 0.5, device=device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MAX_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fused=True,  # bitta CUDA kernel bilan butun optimizer step
    )
    steps_per_epoch = train_split[0].shape[0] // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LEARNING_RATE,
        total_steps=EPOCHS * steps_per_epoch,
        pct_start=0.2,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    generator = torch.Generator(device=device)
    generator.manual_seed(RANDOM_STATE)

    val_actual = val_split[2].cpu().numpy().astype(np.int8)
    best_pr_auc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    epoch_losses: list[float] = []
    training_seconds = 0.0

    for epoch in range(EPOCHS):
        start_event, end_event = _cuda_event_pair()
        start_event.record()
        epoch_loss = train_one_epoch(
            model, optimizer, scheduler, scaler, loss_function,
            train_split, amp_dtype, generator,
        )
        end_event.record()
        end_event.synchronize()
        epoch_seconds = start_event.elapsed_time(end_event) / 1000.0
        training_seconds += epoch_seconds
        average_loss = float(epoch_loss.item())
        epoch_losses.append(average_loss)

        val_probabilities = predict_probabilities(model, val_split, amp_dtype)
        val_pr_auc = float(
            average_precision_score(val_actual, val_probabilities.cpu().numpy())
        )
        if val_pr_auc > best_pr_auc:
            best_pr_auc, best_epoch = val_pr_auc, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(
            f"Epoch {epoch + 1}/{EPOCHS}: loss={average_loss:.5f}, "
            f"val_pr_auc={val_pr_auc:.5f}, vaqt={epoch_seconds:.2f} s"
        )

    model.load_state_dict(best_state)
    print(f"Eng yaxshi epoch: {best_epoch} (val PR-AUC={best_pr_auc:.5f})")

    # Threshold validation'da tanlanadi, test'da faqat bir marta qo'llanadi.
    val_probabilities = predict_probabilities(model, val_split, amp_dtype).cpu().numpy()
    threshold = find_best_threshold(val_actual, val_probabilities)

    start_event, end_event = _cuda_event_pair()
    start_event.record()
    test_probabilities_gpu = predict_probabilities(model, test_split, amp_dtype)
    end_event.record()
    end_event.synchronize()
    evaluation_seconds = start_event.elapsed_time(end_event) / 1000.0

    test_actual = test_split[2].cpu().numpy().astype(np.int8)
    test_probabilities = test_probabilities_gpu.cpu().numpy()
    metrics = compute_metrics(test_actual, test_probabilities, threshold)
    metrics_at_half = compute_metrics(test_actual, test_probabilities, 0.5)

    total_seconds = perf_counter() - total_start
    training_rows = steps_per_epoch * BATCH_SIZE * EPOCHS
    gpu_properties = torch.cuda.get_device_properties(device)

    artifact: dict[str, object] = {
        "format_version": 2,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": {
            "raw_numeric_features": list(RAW_NUMERIC_COLUMNS),
            "feature_names": list(FEATURE_NAMES),
            "categorical_feature": CATEGORICAL_COLUMN,
            "categories": known_categories,
            "hidden_layers": list(HIDDEN_LAYERS),
            "embedding_dimension": EMBEDDING_DIM,
            "dropout": DROPOUT,
        },
        "normalization": {
            "means": prepared["means"],
            "standard_deviations": prepared["stds"],
        },
        "threshold": threshold,
        "metrics": metrics,
        "metrics_at_threshold_0.5": metrics_at_half,
        "training": {
            "device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "amp_dtype": str(amp_dtype),
            "epochs": EPOCHS,
            "best_epoch": best_epoch,
            "batch_size": BATCH_SIZE,
            "max_learning_rate": MAX_LEARNING_RATE,
            "train_rows": int(train_split[0].shape[0]),
            "val_rows": int(val_split[0].shape[0]),
            "test_rows": int(test_split[0].shape[0]),
            "epoch_losses": epoch_losses,
            "data_preparation_seconds": data_seconds,
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "total_seconds": total_seconds,
            "training_rows_per_second": training_rows / training_seconds,
            "evaluation_rows_per_second": test_split[0].shape[0] / evaluation_seconds,
            "gpu_memory_bytes": gpu_properties.total_memory,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        },
    }
    joblib.dump(artifact, RESULT_PATH)

    print(f"\nTest metrikalari (threshold={threshold:.4f}, val'da tanlangan):")
    for name, value in metrics.items():
        print(f"  {name}: {value}")
    print("Taqqoslash uchun threshold=0.5 da:")
    for name in ("precision", "recall", "f1_score"):
        print(f"  {name}: {metrics_at_half[name]}")
    print(
        "\nVaqt: "
        f"o'qitish={training_seconds:.2f} s, "
        f"baholash={evaluation_seconds:.3f} s, "
        f"jami={total_seconds:.2f} s"
    )
    print(
        f"Tezlik: {artifact['training']['training_rows_per_second']:.0f} "  # type: ignore[index]
        "train-row/s"
    )
    print(
        f"GPU xotira (peak): {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB "
        f"/ {gpu_properties.total_memory / 1e9:.2f} GB"
    )
    print(f"Natija saqlandi: {RESULT_PATH}")
    return artifact


if __name__ == "__main__":
    train_model()