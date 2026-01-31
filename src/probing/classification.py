from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from src.classifiers.lor_grid_search_classifier import LoRGridSearchClassifier
from src.probing.config import ClassifyMode


def _load_xy(data_dir: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    x_path = Path(data_dir) / f"X_{split}.npy"
    y_path = Path(data_dir) / f"y_{split}.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing labels file: {y_path}")
    X = np.load(str(x_path))
    y = np.load(str(y_path))
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Shape mismatch for split={split}: X={X.shape}, y={y.shape}")
    return X, y


def _auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    unique = np.unique(y_true)
    if len(unique) <= 2:
        # binary
        return float(roc_auc_score(y_true, y_prob))
    # multiclass
    return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))


def run_classification(
    *,
    data_dir: str,
    model_dir: str,
    clf: str,
    classify_mode: ClassifyMode,
    eval_accuracy: bool,
    hyperparams_yml: str | None,
) -> None:
    data_dir_p = Path(data_dir)
    model_dir_p = Path(model_dir)
    model_dir_p.mkdir(parents=True, exist_ok=True)

    if clf != "lor":
        raise ValueError("Only clf='lor' is currently supported in this repo")

    X_train, y_train = _load_xy(str(data_dir_p), "train")
    X_valid, y_valid = _load_xy(str(data_dir_p), "valid")

    num_classes = len(np.unique(y_train))
    num_classes_for_clf = 1 if num_classes <= 2 else num_classes

    model = LoRGridSearchClassifier(
        hyperparams=hyperparams_yml,
        num_classes=num_classes_for_clf,
    )

    model_path = str(model_dir_p / f"{clf}_best_model")

    if classify_mode == ClassifyMode.INFER_ONLY:
        model.load_best_model(model_path)
    elif classify_mode == ClassifyMode.TRAIN_AND_INFER:
        model.fit(X_train, y_train)
        model.save_best_model(model_path)
    else:
        raise ValueError(f"Unsupported classify_mode={classify_mode}")

    def eval_split(split_name: str, X: np.ndarray, y: np.ndarray) -> None:
        probs = model.predict_probs(X)
        auc = _auc(y, probs)
        print(f"{split_name} AUC: {auc:.4f}")
        if eval_accuracy:
            preds = model.predict(X)
            acc = float(accuracy_score(y, preds))
            print(f"{split_name} Accuracy: {acc:.4f}")

    print("----- Results -----")
    eval_split("train", X_train, y_train)
    eval_split("valid", X_valid, y_valid)

    # Optional test
    x_test_path = data_dir_p / "X_test.npy"
    y_test_path = data_dir_p / "y_test.npy"
    if x_test_path.exists() and y_test_path.exists():
        X_test, y_test = _load_xy(str(data_dir_p), "test")
        eval_split("test", X_test, y_test)

