from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.layer_embedding_extractor import LayerEmbeddingExtractor
from src.probing.config import CreationState, load_embedding_config
from src.utils.file_utils import call_function_by_path


def _prepare_df_for_embedding(
    df: pd.DataFrame,
    *,
    creation_state: CreationState,
    prompt_template_func_path: str | None,
    base_text_column: str,
) -> tuple[pd.DataFrame, str]:
    """
    Returns: (possibly-mutated df, column_name_to_embed)
    """
    if creation_state == CreationState.DATA_ONLY:
        return df, base_text_column

    if creation_state == CreationState.PREDICTION_PROMPT:
        if not prompt_template_func_path:
            raise ValueError("prompt_template_func_path is required for prediction_prompt")

        def to_prompt(example_row: pd.Series) -> str:
            # Convert row to a plain dict for template convenience
            example = example_row.to_dict()
            return call_function_by_path(
                prompt_template_func_path,
                example=example,
                is_train=False,
            )

        df = df.copy()
        df["text"] = df.apply(to_prompt, axis=1)
        return df, "text"

    raise ValueError(f"Unsupported creation_state={creation_state}")


def run_embedding(
    *,
    config_yaml: str,
    out_dir: str,
    layer: int,
    force_save_labels: bool = False,
) -> None:
    cfg = load_embedding_config(config_yaml)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    should_save_labels = True if force_save_labels else cfg.should_save_labels
    if not should_save_labels:
        raise ValueError(
            "Embedding config has should_save_labels=false, but classification requires labels. "
            "Set should_save_labels=true in the YAML or use the pipeline (it forces label saving)."
        )

    extractor = LayerEmbeddingExtractor(cfg.model_path_or_id)

    train_df = pd.read_parquet(cfg.data_train)
    valid_df = pd.read_parquet(cfg.data_valid)

    train_df, train_col = _prepare_df_for_embedding(
        train_df,
        creation_state=cfg.creation_state,
        prompt_template_func_path=cfg.prompt_template_func_path,
        base_text_column=cfg.column_to_embed,
    )
    valid_df, valid_col = _prepare_df_for_embedding(
        valid_df,
        creation_state=cfg.creation_state,
        prompt_template_func_path=cfg.prompt_template_func_path,
        base_text_column=cfg.column_to_embed,
    )

    if train_col != valid_col:
        raise RuntimeError(
            f"Internal error: embed columns mismatch (train={train_col}, valid={valid_col})"
        )

    embed_col = train_col

    extractor.build_embeddings_dataset_at_layer(
        layer_num=layer,
        df=train_df,
        set_type="train",
        out_dir=str(out_path),
        should_save_labels=should_save_labels,
        column=embed_col,
        label_column=cfg.label_column,
    )
    extractor.build_embeddings_dataset_at_layer(
        layer_num=layer,
        df=valid_df,
        set_type="valid",
        out_dir=str(out_path),
        should_save_labels=should_save_labels,
        column=embed_col,
        label_column=cfg.label_column,
    )

    if cfg.data_test:
        test_df = pd.read_parquet(cfg.data_test)
        test_df, test_col = _prepare_df_for_embedding(
            test_df,
            creation_state=cfg.creation_state,
            prompt_template_func_path=cfg.prompt_template_func_path,
            base_text_column=cfg.column_to_embed,
        )
        if test_col != embed_col:
            raise RuntimeError(
                f"Internal error: embed columns mismatch (test={test_col}, expected={embed_col})"
            )
        extractor.build_embeddings_dataset_at_layer(
            layer_num=layer,
            df=test_df,
            set_type="test",
            out_dir=str(out_path),
            should_save_labels=should_save_labels,
            column=embed_col,
            label_column=cfg.label_column,
        )

