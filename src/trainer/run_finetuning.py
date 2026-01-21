"""
Finetuning Module

Trains a model using SFT (Supervised Fine-Tuning) with optional weight injection
for controlled experiments on layer importance and transfer learning.

Supports both CUDA (NVIDIA GPUs) and MPS (Apple Silicon GPUs).
"""

import gc
from typing import Any, Callable, Tuple

import torch
from datasets import Dataset, Split, load_dataset
from peft import LoraConfig
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from src.trainer.weight_injector import WeightInjector
from src.utils import read_yaml, connect_to_hf, set_random_seed
from src.utils import (
    get_device_type,
    get_torch_dtype,
    get_training_precision_args,
    get_optimizer,
    supports_quantization,
    clear_memory_cache,
    print_device_info,
    DeviceType,
)
from src.trainer.trainer_custom_eval import compute_metrics, preprocess_logits_for_metrics, EvaluationArgKey

def finetune_model(
    data_files: dict[Split, str],
    model_output_dir: str,
    prompt_formatting_func: Callable,
    pretrained_model_id: str,
    training_args_yml_path: str,
    injection_params: dict[str, Any] | None,
    evaluation_args_yml_path: str | None = None,
    should_quant_to_4bit: bool = False,
    dataset_type: str = "parquet",
    should_evaluate: bool = True,
    train_set_size: dict[str, float] | None = None,
    seed: int = 42,
    out_cache_dir: str | None = None,
) -> None:
    """
    Train a model using SFT with optional weight injection.

    Args:
        data_files: Dictionary mapping splits to file paths (e.g., {Split.TRAIN: "train.parquet"})
        model_output_dir: Directory for saving trained model and outputs
        prompt_formatting_func: Function to format examples into prompts
        pretrained_model_id: HuggingFace model identifier
        training_args_yml_path: Path to training arguments YAML file
        injection_params: Weight injection configuration or None to disable.
            Expected keys:
                - 'reinit_from_layer': Starting layer for tail reinitialization.
                    If None or not provided: reinitialize ALL layers (full mode)
                    If int: reinitialize from that layer to the end (tail mode)
        evaluation_args_yml_path: Path to evaluation arguments YAML (required if should_evaluate=True)
        should_quant_to_4bit: Whether to use 4-bit quantization (uses LoRA when True)
        dataset_type: Dataset format ("parquet", "csv", etc.)
        should_evaluate: Whether to run evaluation during training
        train_set_size: Dict with 'percentage' (0-1) or 'n' (count) for subset sampling
        seed: Random seed for reproducibility
        out_cache_dir: Optional cache directory for checkpoints
    """
    train_set_size = train_set_size or {'percentage': 1}

    connect_to_hf()
    set_random_seed(seed)

    # Detect and display device info
    device_type = get_device_type()
    print_device_info()

    # Check quantization support
    if should_quant_to_4bit and not supports_quantization(device_type):
        print(f"Warning: 4-bit quantization is not supported on {device_type.value}. Disabling quantization.")
        should_quant_to_4bit = False

    training_args = _load_training_args(training_args_yml_path)
    evaluation_args = _load_evaluation_args(evaluation_args_yml_path) if should_evaluate else {}

    # Load and prepare datasets
    train_dataset, valid_dataset = _prepare_datasets(
        data_files=data_files,
        dataset_type=dataset_type,
        prompt_formatting_func=prompt_formatting_func,
        train_set_size=train_set_size,
        should_evaluate=should_evaluate,
        seed=seed,
    )

    # Load model and tokenizer
    model, tokenizer, peft_config = _load_model_and_tokenizer(
        pretrained_model_id=pretrained_model_id,
        should_quant_to_4bit=should_quant_to_4bit,
        training_args=training_args,
    )

    # Apply weight injection if configured
    if injection_params:
        _apply_weight_injection(model, injection_params)

    # Build training configuration
    sft_config = _build_sft_config(
        model_output_dir=model_output_dir,
        training_args=training_args,
        should_quant_to_4bit=should_quant_to_4bit,
        should_evaluate=should_evaluate,
        seed=seed,
        out_cache_dir=out_cache_dir,
    )

    # Create metric functions
    compute_metrics_fn, preprocess_logits_fn = _create_metric_functions(
        evaluation_args, should_evaluate
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_fn,
        preprocess_logits_for_metrics=preprocess_logits_fn,
    )

    trainer.train()

    # Cleanup
    _cleanup(model, trainer, train_dataset, valid_dataset, should_evaluate)


def _load_training_args(yml_path: str | None) -> dict[str, Any]:
    """Load training arguments from YAML file."""
    if yml_path is None:
        return {}
    return read_yaml(yml_path).get('args', {})


def _load_evaluation_args(yml_path: str | None) -> dict[str, Any]:
    """Load and validate evaluation arguments from YAML file."""
    if yml_path is None:
        return {}

    config = read_yaml(yml_path)
    required_keys = [
        EvaluationArgKey.LAST_PROMPT_TOKEN_ID.value,
        EvaluationArgKey.POSITIVE_TOKEN_ID.value,
        EvaluationArgKey.NEGATIVE_TOKEN_ID.value,
    ]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Evaluation config missing required key: {key}")

    return config


def _prepare_datasets(
    data_files: dict[Split, str],
    dataset_type: str,
    prompt_formatting_func: Callable,
    train_set_size: dict[str, float],
    should_evaluate: bool,
    seed: int,
) -> Tuple[Dataset, Dataset | None]:
    """Load, slice, and format datasets."""

    def add_prompt(example):
        example['text'] = prompt_formatting_func(example)
        return example

    # Load training data
    train_dataset = load_dataset(dataset_type, data_files=data_files, split=Split.TRAIN)
    train_dataset = _slice_dataset(train_dataset, train_set_size, seed)
    train_dataset = train_dataset.map(add_prompt)
    print(f"Train dataset size: {len(train_dataset)}")

    # Load validation data if needed
    valid_dataset = None
    if should_evaluate:
        valid_dataset = load_dataset(dataset_type, data_files=data_files, split=Split.VALIDATION)
        valid_dataset = valid_dataset.map(add_prompt)
        print(f"Validation dataset size: {len(valid_dataset)}")

    return train_dataset, valid_dataset


def _slice_dataset(
    dataset: Dataset,
    train_set_size: dict[str, float],
    seed: int,
) -> Dataset:
    """Optionally slice dataset by percentage or count while preserving label distribution."""
    percentage = train_set_size.get('percentage', -1)
    n = train_set_size.get('n', -1)

    if percentage == 1 or (percentage == -1 and n == -1):
        return dataset

    total_size = len(dataset)

    if 0 < percentage < 1:
        subset_size = int(total_size * percentage)
        print(f"Slicing training set: {percentage*100:.0f}% = {subset_size} samples")
    elif n > 1:
        subset_size = min(int(n), total_size)
        print(f"Slicing training set: n={subset_size} samples")
    else:
        raise ValueError("train_set_size must specify 'percentage' in (0,1] or 'n' > 1")

    indices = list(range(total_size))
    selected_indices, _ = train_test_split(
        indices,
        train_size=subset_size,
        stratify=dataset['label'],
        random_state=seed,
    )

    return dataset.select(selected_indices)


def _load_model_and_tokenizer(
    pretrained_model_id: str,
    should_quant_to_4bit: bool,
    training_args: dict[str, Any],
) -> Tuple[Any, Any, LoraConfig | None]:
    """Load model and tokenizer with optional quantization and PEFT."""
    device_type = get_device_type()
    torch_dtype = get_torch_dtype(device_type)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
    ) if should_quant_to_4bit else None

    peft_config = None

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_id,
        device_map="auto",
        torch_dtype=torch_dtype,
        quantization_config=bnb_config if should_quant_to_4bit else None,
        attn_implementation="eager",
        use_safetensors=True,
    )
    print(f"Loaded model {'with 4-bit quantization' if should_quant_to_4bit else ''} (dtype: {torch_dtype})")

    # Create PEFT config for training with quantization
    if should_quant_to_4bit:
        peft_config = LoraConfig(
            lora_alpha=training_args.get('lora_alpha', 128),
            r=training_args.get('lora_rank', 256),
            lora_dropout=training_args.get('lora_dropout', 0.05),
            bias="none",
            target_modules=training_args.get('target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
            task_type="CAUSAL_LM",
        )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    model.config.pretraining_tp = 1

    return model, tokenizer, peft_config


def _apply_weight_injection(model: Any, injection_params: dict[str, Any]) -> None:
    """
    Apply weight injection to reinitialize layers.

    Args:
        model: The model to reinitialize
        injection_params: Dict with optional 'reinit_from_layer' key.
            - If 'reinit_from_layer' is None or not provided: reinitialize ALL layers
            - If 'reinit_from_layer' is an int: reinitialize from that layer to the end (tail)
    """
    reinit_from_layer = injection_params.get('reinit_from_layer')

    injector = WeightInjector(reinit_from_layer=reinit_from_layer)
    injector.inject(model)

    # Log reinitialized parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}")


def _build_sft_config(
    model_output_dir: str,
    training_args: dict[str, Any],
    should_quant_to_4bit: bool,
    should_evaluate: bool,
    seed: int,
    out_cache_dir: str | None,
) -> SFTConfig:
    """Build SFTConfig from training arguments."""
    output_dir = f"{out_cache_dir}/{model_output_dir}" if out_cache_dir else model_output_dir

    # Get device-appropriate settings
    device_type = get_device_type()
    precision_args = get_training_precision_args(device_type)
    default_optimizer = get_optimizer(device_type)

    # TF32 is only enabled on CUDA when using quantization
    use_tf32 = precision_args['tf32'] and should_quant_to_4bit

    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=training_args.get('num_train_epochs', 5),
        per_device_train_batch_size=training_args.get('per_device_train_batch_size', 3),
        per_device_eval_batch_size=training_args.get('per_device_eval_batch_size', 4),
        gradient_accumulation_steps=training_args.get('gradient_accumulation_steps', 2),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs=training_args.get(
            'gradient_checkpointing_kwargs', {'use_reentrant': False}
        ),
        optim=training_args.get('optim', default_optimizer),
        logging_steps=training_args.get('logging_steps', 100),
        save_strategy=training_args.get('eval_strategy', 'epoch'),
        save_steps=training_args.get('eval_steps'),
        eval_strategy=training_args.get('eval_strategy', 'epoch') if should_evaluate else 'no',
        eval_steps=training_args.get('eval_steps') if should_evaluate else None,
        learning_rate=float(training_args.get('learning_rate', 2e-4)),
        bf16=precision_args['bf16'],
        fp16=precision_args['fp16'],
        tf32=use_tf32,
        max_grad_norm=training_args.get('max_grad_norm', 0.3),
        warmup_ratio=training_args.get('warmup_ratio', 0.03),
        weight_decay=training_args.get('weight_decay', 0),
        lr_scheduler_type=training_args.get('lr_scheduler_type', 'constant'),
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        save_total_limit=training_args.get('save_total_limit'),
        load_best_model_at_end=should_evaluate,
        report_to="tensorboard",
        seed=seed,
        data_seed=seed,
        ddp_find_unused_parameters=False,
        dataset_text_field='text',
        max_length=training_args.get('max_seq_length', 1024),
        packing=False,
        save_safetensors=True,
    )


def _create_metric_functions(
    evaluation_args: dict[str, Any],
    should_evaluate: bool,
) -> Tuple[Callable | None, Callable | None]:
    """Create compute_metrics and preprocess_logits functions."""
    if not should_evaluate:
        return None, None

    def _compute_metrics(eval_pred: Tuple[torch.Tensor, torch.Tensor]):
        return compute_metrics(
            eval_pred,
            last_prompt_token_id=evaluation_args.get(EvaluationArgKey.LAST_PROMPT_TOKEN_ID.value),
            positive_token_id=evaluation_args.get(EvaluationArgKey.POSITIVE_TOKEN_ID.value),
            negative_token_id=evaluation_args.get(EvaluationArgKey.NEGATIVE_TOKEN_ID.value),
            pred_threshold=evaluation_args.get(EvaluationArgKey.PREDICTION_THRESHOLD.value, 0.5),
        )

    def _preprocess_logits(logits: torch.Tensor, labels: torch.Tensor):
        return preprocess_logits_for_metrics(
            logits,
            labels,
            last_prompt_token_id=evaluation_args.get(EvaluationArgKey.LAST_PROMPT_TOKEN_ID.value),
            k=evaluation_args.get(EvaluationArgKey.K.value, 1000),
        )

    return _compute_metrics, _preprocess_logits


def _cleanup(
    model: Any,
    trainer: SFTTrainer,
    train_dataset: Dataset,
    valid_dataset: Dataset | None,
    should_evaluate: bool,
) -> None:
    """Clean up resources after training."""
    del model
    del trainer
    del train_dataset
    if should_evaluate and valid_dataset:
        del valid_dataset

    clear_memory_cache()
    gc.collect()
