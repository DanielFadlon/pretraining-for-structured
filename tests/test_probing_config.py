import textwrap

import pytest

from src.probing.config import CreationState, load_embedding_config, load_run_config


def test_load_embedding_config_minimal(tmp_path):
    cfg = tmp_path / "embed.yaml"
    cfg.write_text(
        textwrap.dedent(
            """\
            model_path_or_id: some/model
            data:
              train: train.parquet
              valid: valid.parquet
            """
        ),
        encoding="utf-8",
    )

    parsed = load_embedding_config(str(cfg))
    assert parsed.model_path_or_id == "some/model"
    assert parsed.data_train == "train.parquet"
    assert parsed.data_valid == "valid.parquet"
    assert parsed.data_test is None
    assert parsed.column_to_embed == "text"
    assert parsed.label_column == "label"
    assert parsed.should_save_labels is True
    assert parsed.creation_state == CreationState.DATA_ONLY
    assert parsed.prompt_template_func_path is None


def test_load_embedding_config_requires_model(tmp_path):
    cfg = tmp_path / "embed.yaml"
    cfg.write_text(
        textwrap.dedent(
            """\
            data:
              train: train.parquet
              valid: valid.parquet
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model_path_or_id"):
        load_embedding_config(str(cfg))


def test_load_embedding_config_prediction_prompt_requires_template(tmp_path):
    cfg = tmp_path / "embed.yaml"
    cfg.write_text(
        textwrap.dedent(
            """\
            model_path_or_id: some/model
            creation_state: prediction_prompt
            data:
              train: train.parquet
              valid: valid.parquet
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prompt_template_func_path"):
        load_embedding_config(str(cfg))


def test_load_run_config_requires_mapping(tmp_path):
    cfg = tmp_path / "run.yaml"
    cfg.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_run_config(str(cfg))

