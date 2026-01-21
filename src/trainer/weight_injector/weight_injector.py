"""
Weight Injector Module

Reinitializes selected layers in a model with random weights.
"""

from typing import List
import torch

class WeightInjector:
    """
    Reinitializes weights of selected layers in a model.

    Args:
        layers_names: List of layer indices to reinitialize (e.g., ['0', '1', '15'])
        use_same_std: If True, use each parameter's original STD for reinitialization.
                      If False, use a constant DEFAULT_STD (0.02).
    """

    def __init__(
        self,
        layers_names: List[str]
    ) -> None:
        if not layers_names:
            raise ValueError("layers_names must be a non-empty list of layer indices")

        self.layers_names = layers_names

    def inject(self, model) -> None:
        """Reinitialize the selected layers in the model."""
        self._reinitialize_selected_layers(model)

    def _reinitialize_selected_layers(self, model) -> None:
        """
        Reinitializes specific layers in the model by their indices.

        Assumes each layer has the following modules:
            - self_attention: q_proj, k_proj, v_proj, o_proj
            - mlp: gate_proj, up_proj, down_proj
            - input_layernorm
            - post_attention_layernorm
        """
        selected_modules = self._build_module_mapping()
        selected_module_names = set(selected_modules.keys())

        for name, module in model.named_modules():
            if name in selected_module_names:
                print(f"Reinitializing layer: {name}")
                self._reinitialize_module(module)

    def _build_module_mapping(self) -> dict:
        """Build mapping of full module paths to layer names."""
        model_block = "model.layers"
        layer_modules = [
            'self_attn.q_proj',
            'self_attn.k_proj',
            'self_attn.v_proj',
            'self_attn.o_proj',
            'mlp.gate_proj',
            'mlp.up_proj',
            'mlp.down_proj',
            'input_layernorm',
            'post_attention_layernorm',
        ]

        mapping = {}
        for layer_name in self.layers_names:
            for module_name in layer_modules:
                full_path = f'{model_block}.{layer_name}.{module_name}'
                mapping[full_path] = layer_name

        return mapping

    def _reinitialize_module(self, module) -> None:
        """Reinitialize weights and biases of a module."""
        if hasattr(module, 'weight') and module.weight is not None:
            self._reinitialize_tensor(module.weight)

        if hasattr(module, 'bias') and module.bias is not None:
            # Biases are always initialized to zeros
            if module.bias.dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
                torch.nn.init.zeros_(module.bias)

    def _reinitialize_tensor(self, tensor: torch.Tensor) -> None:
        """Reinitialize a tensor with normal distribution."""
        supported_dtypes = [torch.float32, torch.float64, torch.float16, torch.bfloat16]

        if tensor.dtype in supported_dtypes:
            std = tensor.data.std().item()
            torch.nn.init.normal_(tensor, mean=0.0, std=std)
        elif tensor.dtype == torch.int8 and hasattr(tensor, 'data_fp32'):
            # Handle quantized int8 weights
            std = tensor.data_fp32.std().item()
            torch.nn.init.normal_(tensor.data_fp32, mean=0.0, std=std)
