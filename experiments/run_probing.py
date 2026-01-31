"""
Run probing: embeddings + linear probe classifier.

This script is the unified entry-point for:
- Creating layer embeddings for train/valid/test splits
- Training (or loading) a classifier on saved embeddings
- Running an end-to-end pipeline

Embeddings output format:
  - X_train.npy, y_train.npy
  - X_valid.npy, y_valid.npy
  - X_test.npy, y_test.npy (optional)

Config schema for embedding (YAML):
  model_path_or_id: "<hf model id or local path>"
  creation_state: "data_only" | "prediction_prompt"
  prompt_template_func_path: "src.templates.binary_classification.prompt_template"  # required for prediction_prompt
  data:
    train: "/path/to/train.parquet"
    valid: "/path/to/valid.parquet"
    test: "/path/to/test.parquet"   # optional
  column_to_embed: "text"           # optional, default: "text"
  label_column: "label"             # optional, default: "label"
  should_save_labels: true          # strongly recommended; required for classification

Execution definition config (YAML) for this script itself:
  command: "pipeline" | "embed" | "classify"   # required
  embed_config: "path/to/embedding_config.yaml"  # required for embed/pipeline
  layer: 15                                      # required for embed/pipeline

  # Embeddings handling (pipeline only)
  embeddings_dir: "output/embeddings"            # optional; if omitted uses temp dir
  keep_embeddings: false                         # optional; only relevant when embeddings_dir is omitted

  # Classification
  model_dir: "output/probe_model"                # required for classify/pipeline
  data_dir: "output/embeddings"                  # required for classify only
  clf: "lor"                                     # optional
  classify_mode: "train_and_infer" | "infer_only"  # optional
  eval_accuracy: true                            # optional
  hyperparams_yml: null                          # optional (defaults inside classifier)
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.probing.pipeline import run_probing_pipeline



def main() -> None:
    parser = argparse.ArgumentParser(description="Run probing based on a YAML config.")
    parser.add_argument(
        "--run-config",
        required=True,
        help="YAML defining execution (see module docstring / configs/examples/probing_run.yaml)",
    )
    args = parser.parse_args()

    run_probing_pipeline(args.run_config)


if __name__ == "__main__":
    main()
