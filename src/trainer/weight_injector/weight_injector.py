"""
Weight Injector Module

Reinitializes selected components of a model with random weights.

Supports two strategies:
    - LAYERS: Reinitialize transformer layers from a starting index (default 0 = all layers)
    - FULL: Reinitialize ALL modules in the model using model.apply()
"""

from enum import Enum
from typing import List
import torch


class ReinitStrategy(str, Enum):
    """Reinitialization strategies."""
    LAYERS = "layers"  # Reinitialize specific transformer layers
    FULL = "full"      # Reinitialize ALL modules via model.apply()


class WeightInjector:
    """
    Reinitializes weights of selected components in a model.

    Strategies:
        - LAYERS: Reinitialize transformer layers from `reinit_from_layer` to end.
          If reinit_from_layer=0 (default), all transformer layers are reinitialized.

        - FULL: Uses model.apply() to reinitialize ALL modules in the model.
          Each weight tensor is reinitialized with normal distribution using the same std.

    Args:
        strategy: ReinitStrategy (LAYERS or FULL)
        reinit_from_layer: Starting layer index (0-based). Default 0 means all layers.
                          Only used when strategy=LAYERS.
    """

    def __init__(
        self,
        strategy: ReinitStrategy,
        reinit_from_layer: int = 0,
    ) -> None:
        self.strategy = strategy
        self.reinit_from_layer = reinit_from_layer
        self.layers_names: List[str] = []

    def inject(self, model) -> None:
        """Reinitialize the selected components in the model."""
        if self.strategy == ReinitStrategy.FULL:
            print("Strategy: FULL (reinitialize all modules via model.apply)")
            model.apply(self._reinitialize_module)
        elif self.strategy == ReinitStrategy.LAYERS:
            num_layers = self._get_num_layers(model)
            self.layers_names = self._calculate_layers_to_reinit(num_layers)

            print(f"Strategy: LAYERS")
            print(f"Total model layers: {num_layers}")
            print(f"Reinitializing transformer layers: {self.layers_names}")

            self._reinitialize_selected_layers(model)

    def _get_num_layers(self, model) -> int:
        """Get the number of transformer layers from the model."""
        if hasattr(model, 'config'):
            if hasattr(model.config, 'num_hidden_layers'):
                return model.config.num_hidden_layers
            if hasattr(model.config, 'n_layer'):
                return model.config.n_layer

        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return len(model.model.layers)

        raise ValueError("Could not determine number of layers in model")

    def _calculate_layers_to_reinit(self, num_layers: int) -> List[str]:
        """Calculate which layers to reinitialize (from reinit_from_layer to end)."""
        start_layer = self.reinit_from_layer

        if start_layer == 0:
            print(f"Reinitializing ALL transformer layers (0 to {num_layers - 1})")
        else:
            print(f"Reinitializing layers {start_layer} to {num_layers - 1}")

        return [str(i) for i in range(start_layer, num_layers)]

    def _reinitialize_selected_layers(self, model) -> None:
        """
        Reinitializes specific transformer layers in the model by their indices.

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
                print(f"Reinitializing: {name}")
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
        """
        Reinitialize weights and biases of a module.
        Uses same std as original weights.
        """
        if hasattr(module, 'weight') and module.weight is not None:
            self._reinitialize_tensor(module.weight)

        if hasattr(module, 'bias') and module.bias is not None:
            self._reinitialize_tensor(module.bias)

    def _reinitialize_tensor(self, tensor: torch.Tensor) -> None:
        """
        Reinitialize a tensor with normal distribution using same std as original.
        """
        if not torch.is_floating_point(tensor):
            return

        param_std = tensor.data.std().item()

        with torch.no_grad():
            torch.nn.init.normal_(tensor, mean=0.0, std=param_std)
