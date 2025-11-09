"""
eVTOL Reinforcement Learning Package
------------------------------------
Contains all modules for training autonomous eVTOL flight agents using PPO.

Modules:
- env.py     → Simulation / AirSim environment wrapper
- model.py   → Actor-Critic neural network for PPO
- utils.py   → Utilities: seeding, checkpointing, logging
- train_rl.py → PPO training script

This __init__.py ensures reproducibility and simple package imports.
"""

import os
import sys
import torch
import random
import numpy as np

# Make sure the root directory is on PYTHONPATH for imports
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

__all__ = ["env", "model", "utils", "train_rl"]


def initialize_environment(seed: int = 42, deterministic: bool = False):
    """
    Set global seeds for reproducibility across PyTorch, NumPy, and Python.

    Args:
        seed (int): Random seed.
        deterministic (bool): If True, enables deterministic CUDA ops.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[INFO] Environment initialized with seed {seed} | Deterministic={deterministic}")
