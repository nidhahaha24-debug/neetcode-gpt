import torch
import torch.nn as nn
import math
from typing import List


class Solution:
    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        W = torch.randn(fan_out, fan_in) * std
        return [[round(v, 4) for v in row] for row in W.tolist()]

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        W = torch.randn(fan_out, fan_in) * std
        return [[round(v, 4) for v in row] for row in W.tolist()]

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)
        t = init_type.lower()
        def _std(f_in: int, f_out: int) -> float:
            if "kaiming" in t or t == "he" or "he_" in t or "relu" in t:
                return math.sqrt(2.0 / f_in)
            if "xavier" in t or "glorot" in t:
                return math.sqrt(2.0 / (f_in + f_out))
            # default / standard / none -> std 1.0 (no scaling)
            return 1.0
        weights = []
        weights.append(torch.randn(hidden_dim, input_dim) * _std(input_dim, hidden_dim))
        for _ in range(1, num_layers):
            weights.append(torch.randn(hidden_dim, hidden_dim) * _std(hidden_dim, hidden_dim))
        x = torch.randn(1, input_dim)
        out: List[float] = []
        for W in weights:
            x = torch.relu(x @ W.T)
            out.append(round(x.std().item(), 2))
        return out