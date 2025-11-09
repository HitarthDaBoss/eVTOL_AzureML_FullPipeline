import os
import json
import yaml
import torch
import random
import numpy as np
from datetime import datetime

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic settings (optional; may slow training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def save_checkpoint(state: dict, path: str):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)
    print(f"[utils] Saved checkpoint: {path}")

def load_checkpoint(path: str, map_location='cpu'):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location=map_location)

class RunningMeanStd:
    """
    Welford's algorithm for online mean/std (for observation normalization).
    Maintains running mean and variance; mergeable if needed.
    """
    def __init__(self, eps=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = eps

    def update(self, x: np.ndarray):
        x = np.array(x)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * (self.count * batch_count / tot_count)
        self.mean = new_mean
        self.var = M2 / (tot_count)
        self.count = tot_count

    def normalize(self, x, clip=10.0):
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8), -clip, clip)
