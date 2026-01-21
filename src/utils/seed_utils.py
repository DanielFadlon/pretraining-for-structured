import torch

from transformers import set_seed
import random
import numpy as np

def set_random_seed(seed: int) -> None:
    # Set seed for Hugging Face transformers
    set_seed(seed)

    # Set seed for Python random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
