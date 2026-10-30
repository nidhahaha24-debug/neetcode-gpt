import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        x_arr = np.array(x, dtype=np.float64)
        g_arr = np.array(gamma, dtype=np.float64)
        rms = np.sqrt(np.mean(x_arr ** 2) + eps)
        out = (x_arr / rms) * g_arr
        return np.round(out, 4).tolist()