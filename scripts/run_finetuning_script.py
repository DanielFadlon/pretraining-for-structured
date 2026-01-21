"""
Finetuning Script

Finetune a model with optional weight injection on selected layers.

Usage:
    python run_finetuning_script.py <config.yaml> <output_dir> [cache_dir]
"""

import os
import sys
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Split

from trainer.run_finetuning import finetune_model
from utils.file_utils import call_function_by_path, read_yaml


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
    'injection_params': None,   # None means no injection
    'evaluation_args_yml_path': None,
    'saved_peft_model_id_or_path': None,
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

    # Validate injection_params if provided
    injection_params = config.get('injection_params')
    if injection_params:
        if 'layers_names' not in injection_params:
            raise ValueError("injection_params must include 'layers_names'")
        if not isinstance(injection_params['layers_names'], list):
            raise ValueError("injection_params['layers_names'] must be a list")


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

    # Override cache_dir from CLI if provided
    out_cache_dir = cache_dir or config['out_cache_dir']

    finetune_model(
        data_files=data_files,
        model_output_dir=model_output_dir,
        prompt_formatting_func=prompt_formatter,
        pretrained_model_id=config['pretrained_model_id'],
        training_args_yml_path=config['training_args_yml_path'],
        evaluation_args_yml_path=config['evaluation_args_yml_path'],
        injection_params=config['injection_params'],
        peft_model_id_or_path=config['saved_peft_model_id_or_path'],
        should_quant_to_4bit=config['should_quant_to_4bit'],
        is_continual_training=config['is_continual_training'],
        should_evaluate=should_evaluate,
        train_set_size=config['train_set_size'],
        seed=config['seed'],
        out_cache_dir=out_cache_dir,
    )


def main() -> None:
    """Main entry point."""
    yaml_path, model_output_dir, cache_dir = parse_args()

    print(f"\n{'='*60}")
    print(f"Finetuning Script - PID: {os.getpid()}")
    print(f"{'='*60}")
    print(f"Config:     {yaml_path}")
    print(f"Output dir: {model_output_dir}")
    if cache_dir:
        print(f"Cache dir:  {cache_dir}")
    print(f"{'='*60}\n")

    config = load_config(yaml_path)
    run_training(config, model_output_dir, cache_dir)


if __name__ == "__main__":
    main()
