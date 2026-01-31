import textwrap

import numpy as np

from src.probing.classification import _auc, _load_xy
from src.probing.pipeline import run_probing_pipeline


def test_auc_binary():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    val = _auc(y, p)
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0


def test_auc_multiclass():
    y = np.array([0, 1, 2, 1, 0])
    p = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.05, 0.9],
            [0.1, 0.85, 0.05],
            [0.8, 0.15, 0.05],
        ]
    )
    val = _auc(y, p)
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0


def test_load_xy_roundtrip(tmp_path):
    X = np.arange(12, dtype=np.float32).reshape(3, 4)
    y = np.array([0, 1, 0], dtype=np.int64)
    np.save(tmp_path / "X_train.npy", X)
    np.save(tmp_path / "y_train.npy", y)

    X2, y2 = _load_xy(str(tmp_path), "train")
    assert X2.shape == (3, 4)
    assert y2.shape == (3,)
    assert np.allclose(X2, X)
    assert np.array_equal(y2, y)


def test_run_probing_pipeline_dispatch_tmpdir_deleted(tmp_path, monkeypatch):
    calls = {"embed": None, "classify": None}

    def fake_run_embedding(*, config_yaml: str, out_dir: str, layer: int, force_save_labels: bool):
        calls["embed"] = {
            "config_yaml": config_yaml,
            "out_dir": out_dir,
            "layer": layer,
            "force_save_labels": force_save_labels,
        }
        # Create a sentinel file so we can verify cleanup.
        with open(f"{out_dir}/sentinel.txt", "w", encoding="utf-8") as f:
            f.write("ok")

    def fake_run_classification(
        *, data_dir: str, model_dir: str, clf: str, classify_mode, eval_accuracy: bool, hyperparams_yml
    ):
        calls["classify"] = {
            "data_dir": data_dir,
            "model_dir": model_dir,
            "clf": clf,
        }

    import src.probing.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run_embedding", fake_run_embedding)
    monkeypatch.setattr(pipeline_mod, "run_classification", fake_run_classification)

    run_cfg = tmp_path / "run.yaml"
    run_cfg.write_text(
        textwrap.dedent(
            f"""\
            command: pipeline
            embed_config: {tmp_path}/embed.yaml
            layer: 5
            model_dir: {tmp_path}/model
            keep_embeddings: false
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "embed.yaml").write_text(
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

    run_probing_pipeline(str(run_cfg))

    assert calls["embed"] is not None
    assert calls["classify"] is not None

    tmp_embeddings_dir = calls["embed"]["out_dir"]
    # Should be cleaned up after pipeline when embeddings_dir is omitted and keep_embeddings is false.
    import os

    assert not os.path.exists(tmp_embeddings_dir)


def test_run_probing_pipeline_embed_dispatch(tmp_path, monkeypatch):
    calls = {"embed": None}

    def fake_run_embedding(*, config_yaml: str, out_dir: str, layer: int, force_save_labels: bool):
        calls["embed"] = {
            "config_yaml": config_yaml,
            "out_dir": out_dir,
            "layer": layer,
            "force_save_labels": force_save_labels,
        }

    import src.probing.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run_embedding", fake_run_embedding)

    out_dir = tmp_path / "emb"
    run_cfg = tmp_path / "run.yaml"
    run_cfg.write_text(
        textwrap.dedent(
            f"""\
            command: embed
            embed_config: {tmp_path}/embed.yaml
            layer: 3
            embeddings_dir: {out_dir}
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "embed.yaml").write_text(
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

    run_probing_pipeline(str(run_cfg))
    assert calls["embed"]["out_dir"] == str(out_dir)
    assert calls["embed"]["layer"] == 3
    assert calls["embed"]["force_save_labels"] is False

