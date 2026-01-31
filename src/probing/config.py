from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.utils.file_utils import read_yaml


class CreationState(str, Enum):
    DATA_ONLY = "data_only"
    PREDICTION_PROMPT = "prediction_prompt"


class ClassifyMode(str, Enum):
    INFER_ONLY = "infer_only"
    TRAIN_AND_INFER = "train_and_infer"


class RunCommand(str, Enum):
    PIPELINE = "pipeline"
    EMBED = "embed"
    CLASSIFY = "classify"


@dataclass(frozen=True)
class EmbeddingConfig:
    model_path_or_id: str
    data_train: str
    data_valid: str
    data_test: str | None
    column_to_embed: str
    label_column: str
    should_save_labels: bool
    creation_state: CreationState
    prompt_template_func_path: str | None


def load_embedding_config(yaml_path: str) -> EmbeddingConfig:
    args = read_yaml(yaml_path) or {}
    data_paths = (args.get("data") or {})

    model_path_or_id = args.get("model_path_or_id")
    if not model_path_or_id:
        raise ValueError("Config must specify 'model_path_or_id'")

    data_train = data_paths.get("train")
    data_valid = data_paths.get("valid")
    if not data_train or not data_valid:
        raise ValueError("Config must specify data.train and data.valid parquet paths")

    state_raw = (args.get("creation_state") or CreationState.DATA_ONLY.value)
    try:
        creation_state = CreationState(state_raw)
    except ValueError as e:
        raise ValueError(
            f"Invalid creation_state={state_raw!r}. Must be one of: {[s.value for s in CreationState]}"
        ) from e

    prompt_template_func_path = args.get("prompt_template_func_path")
    if creation_state == CreationState.PREDICTION_PROMPT and not prompt_template_func_path:
        raise ValueError(
            "Config must specify 'prompt_template_func_path' when creation_state is 'prediction_prompt'"
        )

    should_save_labels = bool(args.get("should_save_labels", True))

    return EmbeddingConfig(
        model_path_or_id=str(model_path_or_id),
        data_train=str(data_train),
        data_valid=str(data_valid),
        data_test=str(data_paths["test"]) if data_paths.get("test") else None,
        column_to_embed=str(args.get("column_to_embed", "text")),
        label_column=str(args.get("label_column", "label")),
        should_save_labels=should_save_labels,
        creation_state=creation_state,
        prompt_template_func_path=str(prompt_template_func_path) if prompt_template_func_path else None,
    )


def load_run_config(yaml_path: str) -> dict[str, Any]:
    cfg = read_yaml(yaml_path) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Run config must be a YAML mapping/dict. Got: {type(cfg)}")
    return cfg

