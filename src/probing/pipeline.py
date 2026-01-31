from __future__ import annotations

import shutil
import tempfile
from typing import Any

from src.probing.classification import run_classification
from src.probing.config import ClassifyMode, RunCommand, load_run_config
from src.probing.embedding import run_embedding


def run_probing_pipeline(config: str | dict[str, Any]) -> None:
    """
    Run probing pipeline according to a "run config".

    Args:
        config:
            Either:
            - Path to a run-config YAML file, or
            - An already-loaded dict with the run-config contents.
    """
    cfg = load_run_config(config) if isinstance(config, str) else config

    cmd_raw = cfg.get("command")
    if not cmd_raw:
        raise ValueError("Run config must specify 'command' = 'pipeline'|'embed'|'classify'")
    try:
        cmd = RunCommand(str(cmd_raw))
    except ValueError as e:
        raise ValueError(
            f"Invalid command={cmd_raw!r}. Must be one of: {[c.value for c in RunCommand]}"
        ) from e

    # Shared defaults
    clf = str(cfg.get("clf", "lor"))
    classify_mode = ClassifyMode(str(cfg.get("classify_mode", ClassifyMode.TRAIN_AND_INFER.value)))
    eval_accuracy = bool(cfg.get("eval_accuracy", True))
    hyperparams_yml = cfg.get("hyperparams_yml")
    hyperparams_yml = str(hyperparams_yml) if hyperparams_yml else None

    if cmd in (RunCommand.EMBED, RunCommand.PIPELINE):
        embed_config = cfg.get("embed_config")
        if not embed_config:
            raise ValueError("Run config must specify 'embed_config' for embed/pipeline")
        layer = cfg.get("layer")
        if layer is None:
            raise ValueError("Run config must specify 'layer' for embed/pipeline")
        layer = int(layer)

    if cmd == RunCommand.EMBED:
        embeddings_dir = cfg.get("embeddings_dir")
        if not embeddings_dir:
            raise ValueError("Run config must specify 'embeddings_dir' for embed")
        run_embedding(
            config_yaml=str(embed_config),
            out_dir=str(embeddings_dir),
            layer=layer,
            force_save_labels=False,
        )
        return

    if cmd == RunCommand.CLASSIFY:
        data_dir = cfg.get("data_dir")
        model_dir = cfg.get("model_dir")
        if not data_dir:
            raise ValueError("Run config must specify 'data_dir' for classify")
        if not model_dir:
            raise ValueError("Run config must specify 'model_dir' for classify")
        run_classification(
            data_dir=str(data_dir),
            model_dir=str(model_dir),
            clf=clf,
            classify_mode=classify_mode,
            eval_accuracy=eval_accuracy,
            hyperparams_yml=hyperparams_yml,
        )
        return

    if cmd == RunCommand.PIPELINE:
        model_dir = cfg.get("model_dir")
        if not model_dir:
            raise ValueError("Run config must specify 'model_dir' for pipeline")

        embeddings_dir = cfg.get("embeddings_dir")
        keep_embeddings = bool(cfg.get("keep_embeddings", False))

        tmp_dir: str | None = None
        if embeddings_dir is None:
            tmp_dir = tempfile.mkdtemp(prefix="probing_embeddings_")
            embeddings_dir = tmp_dir

        try:
            run_embedding(
                config_yaml=str(embed_config),
                out_dir=str(embeddings_dir),
                layer=layer,
                force_save_labels=True,
            )
            run_classification(
                data_dir=str(embeddings_dir),
                model_dir=str(model_dir),
                clf=clf,
                classify_mode=classify_mode,
                eval_accuracy=eval_accuracy,
                hyperparams_yml=hyperparams_yml,
            )
        finally:
            if tmp_dir and not keep_embeddings:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    raise RuntimeError(f"Unhandled command: {cmd}")
