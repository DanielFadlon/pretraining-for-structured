"""
Weight Injector Module

Reinitializes selected layers in a model with random weights.
Supports full reinitialization (all layers) or tail reinitialization (from a specific layer to the end).
"""

from typing import List
import torch


class WeightInjector:
    """
    Reinitializes weights of selected layers in a model.

    Modes:
        - Full reinitialization: When reinit_from_layer is None, all layers are reinitialized.
        - Tail reinitialization: When reinit_from_layer is specified, layers from that index
          to the last layer (inclusive) are reinitialized.

    Args:
        reinit_from_layer: Starting layer index for tail reinitialization.
                          If None, all layers will be reinitialized (full mode).
    """

    def __init__(self, reinit_from_layer: int | None = None) -> None:
        self.reinit_from_layer = reinit_from_layer
        self.layers_names: List[str] = []  # Will be populated during inject()

    def inject(self, model) -> None:
        """Reinitialize the selected layers in the model."""
        # Auto-detect number of layers from model
        num_layers = self._get_num_layers(model)
        # Calculate which layers to reinitialize
        self.layers_names = self._calculate_layers_to_reinit(num_layers)

        print(f"Total model layers: {num_layers}")
        print(f"Reinitializing layers: {self.layers_names}")

        self._reinitialize_selected_layers(model)

    def _get_num_layers(self, model) -> int:
        """Get the number of transformer layers from the model."""
        # Try common attribute names for layer count
        if hasattr(model, 'config'):
            if hasattr(model.config, 'num_hidden_layers'):
                return model.config.num_hidden_layers
            if hasattr(model.config, 'n_layer'):
                return model.config.n_layer

        # Fallback: count layers directly
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return len(model.model.layers)

        raise ValueError("Could not determine number of layers in model")

    def _calculate_layers_to_reinit(self, num_layers: int) -> List[str]:
        """
        Calculate which layers to reinitialize.

        - If reinit_from_layer is None: reinitialize all layers (0 to num_layers-1)
        - Otherwise: reinitialize tail (reinit_from_layer to num_layers-1)
        """
        if self.reinit_from_layer is None:
            # Full reinitialization: all layers
            start_layer = 0
            print("Mode: FULL reinitialization (all layers)")
        else:
            # Tail reinitialization: from specified layer to end
            start_layer = self.reinit_from_layer
            print(f"Mode: TAIL reinitialization (from layer {start_layer} to end)")

        return [str(i) for i in range(start_layer, num_layers)]

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
