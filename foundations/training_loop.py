import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        n = X.shape[0]
        w = np.zeros(X.shape[1], dtype=np.float64)
        b = 0.0
        for _ in range(epochs):
            y_hat = X @ w + b
            err = y_hat - y
            dw = (2.0 / n) * (X.T @ err)
            db = (2.0 / n) * np.sum(err)
            w = w - lr * dw
            b = b - lr * db
        return np.round(w, 5), round(float(b), 5)