import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str = "sigmoid") -> float:
        z = float(np.dot(x, w) + b)
        if activation == "sigmoid":
            y = 1.0 / (1.0 + np.exp(-z))
        else:
            y = max(0.0, z)
        return round(float(y), 5)

    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float, activation: str = "sigmoid"):
        z = float(np.dot(x, w) + b)
        if activation == "sigmoid":
            p = 1.0 / (1.0 + np.exp(-z))
            dp_dz = p * (1.0 - p)
        else:
            p = max(0.0, z)
            dp_dz = 1.0 if z > 0 else 0.0
        dL_dz = (p - y_true) * dp_dz
        dw = [round(float(v), 5) for v in (dL_dz * np.array(x))]
        db = round(float(dL_dz), 5)
        return (dw, db)