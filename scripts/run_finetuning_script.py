"""
Finetuning Script

Finetune a model with optional weight reinitialization.

Usage:
    python scripts/run_finetuning_script.py <config.yaml> <output_dir> [cache_dir]

Run Types (set via 'run_type' in config):
    - "regular"          - Standard finetuning without weight reinitialization
    - "reinit_layers"    - Reinitialize transformer layers from 'reinit_from_layer' (default 0 = all layers)
    - "full_reinitialize" - Reinitialize ALL modules via model.apply() (equivalent to training from scratch)
"""

import os
import sys
from enum import Enum
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Split

from src.trainer.run_finetuning import finetune_model
from src.utils.file_utils import call_function_by_path, read_yaml


class RunType(str, Enum):
    """Enum for different finetuning run types."""
    REGULAR = "regular"                    # Standard finetuning, no reinitialization
    REINIT_LAYERS = "reinit_layers"        # Reinitialize transformer layers (from reinit_from_layer, default 0)
    FULL_REINITIALIZE = "full_reinitialize"  # Reinitialize ALL modules via model.apply()


# Required configuration values
REQUIRED_CONFIG_VALUES = [
    'dataset_dir',
    'pretrained_model_id',
    'training_args_yml_path',
    'prompt_template_func_path',
]

# Default configuration values (for optional arguments)
DEFAULTS = {
    'train_data_file_name': 'train.parquet',
    'validation_data_file_name': 'valid.parquet',
    'should_quant_to_4bit': False,
    'run_type': RunType.REGULAR.value,
    'reinit_from_layer': 0,              # Default: reinitialize from layer 0 (all layers)
    'evaluation_args_yml_path': None,
    'should_evaluate': True,
    'train_set_size': {'percentage': 1},
    'seed': 42,
    'out_cache_dir': None,
}


def parse_args() -> tuple[str, str, str | None]:
    """Parse and validate command line arguments."""
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    yaml_path = sys.argv[1]
    model_output_dir = sys.argv[2]
    cache_dir = sys.argv[3] if len(sys.argv) > 3 else None

    return yaml_path, model_output_dir, cache_dir


def validate_config(config: dict[str, Any]) -> None:
    """Validate the configuration."""
    for required_value in REQUIRED_CONFIG_VALUES:
        if required_value not in config:
            raise ValueError(f"Config must specify '{required_value}'")

    # Validate run_type
    run_type_value = config.get('run_type', RunType.REGULAR.value)
    valid_run_types = [rt.value for rt in RunType]
    if run_type_value not in valid_run_types:
        raise ValueError(
            f"run_type must be one of {valid_run_types} "
            f"('regular', 'reinit_layers', 'full_reinitialize')"
        )

    run_type = RunType(run_type_value)

    # Validate reinit_from_layer if provided
    if run_type in [RunType.REINIT_LAYERS]:
        reinit_from_layer = config.get('reinit_from_layer', 0)
        if not isinstance(reinit_from_layer, int) or reinit_from_layer < 0:
            raise ValueError("reinit_from_layer must be a non-negative integer")


def load_config(yaml_path: str) -> dict[str, Any]:
    """Load configuration from YAML file with defaults."""
    config = read_yaml(yaml_path)
    validate_config(config)

    # Apply defaults for missing keys
    for key, default in DEFAULTS.items():
        config.setdefault(key, default)

    return config


def create_prompt_formatter(prompt_template_func_path: str):
    """Create a prompt formatting function for training."""
    def format_prompt(example: dict) -> str:
        return call_function_by_path(
            prompt_template_func_path,
            example=example,
            is_train=True
        )
    return format_prompt


def build_injection_params(config: dict[str, Any]) -> dict[str, Any] | None:
    """
    Build injection_params based on run_type.

    Returns:
        None for REGULAR run type, or dict with reinitialization settings.
    """
    run_type = RunType(config['run_type'])

    if run_type == RunType.REGULAR:
        return None

    if run_type == RunType.REINIT_LAYERS:
        # Reinitialize only transformer layers (from reinit_from_layer to end)
        return {
            'reinit_from_layer': config['reinit_from_layer'],
            'full_reinit': False,
        }

    if run_type == RunType.FULL_REINITIALIZE:
        # Reinitialize ALL modules via model.apply()
        return {
            'full_reinit': True,
        }

    return None


def run_training(config: dict[str, Any], model_output_dir: str, cache_dir: str | None) -> None:
    """Execute the training pipeline."""
    dataset_dir = config['dataset_dir']
    train_file = config['train_data_file_name']
    should_evaluate = config['should_evaluate']

    # Build data files dict
    data_files = {
        Split.TRAIN: os.path.join(dataset_dir, train_file),
    }
    if should_evaluate:
        data_files[Split.VALIDATION] = os.path.join(dataset_dir, config['validation_data_file_name'])

    # Create prompt formatter
    prompt_formatter = create_prompt_formatter(config['prompt_template_func_path'])

    # Build injection params based on run_type
    injection_params = build_injection_params(config)

    # Override cache_dir from CLI if provided
    out_cache_dir = cache_dir or config['out_cache_dir']

    finetune_model(
        data_files=data_files,
        model_output_dir=model_output_dir,
        prompt_formatting_func=prompt_formatter,
        pretrained_model_id=config['pretrained_model_id'],
        training_args_yml_path=config['training_args_yml_path'],
        evaluation_args_yml_path=config['evaluation_args_yml_path'],
        injection_params=injection_params,
        should_quant_to_4bit=config['should_quant_to_4bit'],
        should_evaluate=should_evaluate,
        train_set_size=config['train_set_size'],
        seed=config['seed'],
        out_cache_dir=out_cache_dir,
    )


def main() -> None:
    """Main entry point."""
    yaml_path, model_output_dir, cache_dir = parse_args()

    config = load_config(yaml_path)
    run_type = RunType(config['run_type'])

    print(f"\n{'='*60}")
    print(f"Finetuning Script - PID: {os.getpid()}")
    print(f"{'='*60}")
    print(f"Config:     {yaml_path}")
    print(f"Output dir: {model_output_dir}")
    print(f"Run type:   {run_type.value}")

    if run_type == RunType.REINIT_LAYERS:
        layer = config['reinit_from_layer']
        if layer == 0:
            print(f"Reinit:     All transformer layers (from layer 0)")
        else:
            print(f"Reinit:     Layers {layer} to end")
    elif run_type == RunType.FULL_REINITIALIZE:
        print(f"Reinit:     FULL (all modules via model.apply)")

    if cache_dir:
        print(f"Cache dir:  {cache_dir}")
    print(f"{'='*60}\n")

    run_training(config, model_output_dir, cache_dir)


if __name__ == "__main__":
    main()
