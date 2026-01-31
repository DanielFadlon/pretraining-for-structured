"""
Finetuning Script

Finetune a model with optional weight reinitialization.

Usage:
    python experiments/run_finetuning_script.py <config.yaml> <output_dir> [cache_dir]

Reinit Strategies (set via 'reinit_strategy' in config):
    - null (or omitted) - Standard finetuning without weight reinitialization
    - "layers"          - Reinitialize transformer layers from 'reinit_from_layer' (default 0 = all)
    - "full"            - Reinitialize ALL modules via model.apply()
"""

import os
import sys
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Split

from src.trainer.run_finetuning import finetune_model
from src.trainer.weight_injector import ReinitStrategy
from src.utils.file_utils import call_function_by_path, read_yaml


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
    'reinit_strategy': None,             # None = no reinitialization, "layers" or "full"
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

    # Validate reinit_strategy if provided
    reinit_strategy = config.get('reinit_strategy')
    if reinit_strategy is not None:
        valid_strategies = [s.value for s in ReinitStrategy]
        if reinit_strategy not in valid_strategies:
            raise ValueError(
                f"reinit_strategy must be one of {valid_strategies} or null"
            )

        # Validate reinit_from_layer for LAYERS strategy
        if reinit_strategy == ReinitStrategy.LAYERS.value:
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
    Build injection_params based on reinit_strategy.

    Returns:
        None if no reinitialization, or dict with strategy and settings.
    """
    reinit_strategy = config.get('reinit_strategy')

    if reinit_strategy is None:
        return None

    if reinit_strategy == ReinitStrategy.LAYERS.value:
        return {
            'strategy': ReinitStrategy.LAYERS,
            'reinit_from_layer': config['reinit_from_layer'],
        }

    if reinit_strategy == ReinitStrategy.FULL.value:
        return {
            'strategy': ReinitStrategy.FULL,
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

    # Build injection params based on reinit_strategy
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
    reinit_strategy = config.get('reinit_strategy')

    print(f"\n{'='*60}")
    print(f"Finetuning Script - PID: {os.getpid()}")
    print(f"{'='*60}")
    print(f"Config:     {yaml_path}")
    print(f"Output dir: {model_output_dir}")

    if reinit_strategy is None:
        print(f"Reinit:     None (regular finetuning)")
    elif reinit_strategy == ReinitStrategy.LAYERS.value:
        layer = config['reinit_from_layer']
        if layer == 0:
            print(f"Reinit:     LAYERS (all transformer layers)")
        else:
            print(f"Reinit:     LAYERS (from layer {layer} to end)")
    elif reinit_strategy == ReinitStrategy.FULL.value:
        print(f"Reinit:     FULL (all modules)")

    if cache_dir:
        print(f"Cache dir:  {cache_dir}")
    print(f"{'='*60}\n")

    run_training(config, model_output_dir, cache_dir)


if __name__ == "__main__":
    main()
