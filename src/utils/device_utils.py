"""
Device Utilities Module

Provides utilities for detecting and handling different compute devices (CUDA, MPS, CPU).
"""

import torch
from enum import Enum
from typing import Any


class DeviceType(str, Enum):
    """Supported device types."""
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


def get_device_type() -> DeviceType:
    """
    Detect the best available device.

    Returns:
        DeviceType: CUDA if available, MPS if on Apple Silicon, otherwise CPU.
    """
    if torch.cuda.is_available():
        return DeviceType.CUDA
    elif torch.backends.mps.is_available():
        return DeviceType.MPS
    else:
        return DeviceType.CPU


def get_torch_dtype(device_type: DeviceType | None = None) -> torch.dtype:
    """
    Get the appropriate torch dtype for the device.

    - CUDA: bfloat16 (works with bf16 mixed precision training)
    - MPS: float32 (MPS doesn't support mixed precision with gradient scaling,
           so we use float32 for stable training)
    - CPU: float32

    Args:
        device_type: The device type. If None, auto-detects.

    Returns:
        torch.dtype: bfloat16 for CUDA, float32 for MPS/CPU.
    """
    if device_type is None:
        device_type = get_device_type()

    if device_type == DeviceType.CUDA:
        return torch.bfloat16
    else:
        # MPS and CPU use float32 for stable training
        # (MPS doesn't support mixed precision with gradient scaling)
        return torch.float32


def get_training_precision_args(device_type: DeviceType | None = None) -> dict[str, bool]:
    """
    Get the appropriate precision arguments for training configuration.

    Args:
        device_type: The device type. If None, auto-detects.

    Returns:
        Dict with 'bf16', 'fp16', and 'tf32' settings.

    Note:
        MPS doesn't support FP16 gradient scaling, so we disable mixed precision
        training on MPS. The model itself can still use float16 dtype for memory
        efficiency, but the training loop runs without mixed precision.
    """
    if device_type is None:
        device_type = get_device_type()

    if device_type == DeviceType.CUDA:
        return {
            'bf16': True,
            'fp16': False,
            'tf32': True,
        }
    elif device_type == DeviceType.MPS:
        # MPS doesn't support FP16 gradient scaling (causes "Attempting to unscale FP16 gradients" error)
        # Disable mixed precision training - model still uses float16 dtype for memory efficiency
        return {
            'bf16': False,
            'fp16': False,
            'tf32': False,
        }
    else:
        # CPU fallback
        return {
            'bf16': False,
            'fp16': False,
            'tf32': False,
        }


def get_optimizer(device_type: DeviceType | None = None, default: str = 'adamw_torch') -> str:
    """
    Get the appropriate optimizer for the device.

    Fused optimizers are faster but only work on CUDA.

    Args:
        device_type: The device type. If None, auto-detects.
        default: Fallback optimizer name.

    Returns:
        Optimizer name string.
    """
    if device_type is None:
        device_type = get_device_type()

    if device_type == DeviceType.CUDA:
        return 'adamw_torch_fused'
    else:
        return default


def supports_quantization(device_type: DeviceType | None = None) -> bool:
    """
    Check if the device supports bitsandbytes quantization.

    Currently only CUDA supports bitsandbytes.

    Args:
        device_type: The device type. If None, auto-detects.

    Returns:
        True if quantization is supported.
    """
    if device_type is None:
        device_type = get_device_type()

    return device_type == DeviceType.CUDA


def clear_memory_cache(device_type: DeviceType | None = None) -> None:
    """
    Clear the memory cache for the current device.

    Args:
        device_type: The device type. If None, auto-detects.
    """
    if device_type is None:
        device_type = get_device_type()

    if device_type == DeviceType.CUDA:
        torch.cuda.empty_cache()
    elif device_type == DeviceType.MPS:
        torch.mps.empty_cache()


def print_device_info() -> None:
    """Print information about the detected device."""
    device_type = get_device_type()
    precision_args = get_training_precision_args(device_type)
    torch_dtype = get_torch_dtype(device_type)
    print(f"Device: {device_type.value}")

    if device_type == DeviceType.CUDA:
        print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Training mode: bfloat16 mixed precision")
    elif device_type == DeviceType.MPS:
        print("  Apple Silicon GPU (Metal Performance Shaders)")
        print("  Training mode: float32 (MPS doesn't support mixed precision)")
    else:
        print("  CPU only")
        print("  Training mode: float32")

    print(f"  Model dtype: {torch_dtype}")
    print(f"  Quantization support: {supports_quantization(device_type)}")
