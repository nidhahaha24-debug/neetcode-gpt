import numpy as np


class Solution:
    def forward_and_backward(self, x, W1, b1, W2, b2, y_true):
        x = np.array(x, dtype=np.float64)
        W1 = np.array(W1, dtype=np.float64)
        b1 = np.array(b1, dtype=np.float64)
        W2 = np.array(W2, dtype=np.float64)
        b2 = np.array(b2, dtype=np.float64)
        y_true = np.array(y_true, dtype=np.float64)
        z1 = W1 @ x + b1
        h = np.maximum(0.0, z1)
        y_pred = W2 @ h + b2
        diff = y_pred - y_true
        loss = float(np.mean(diff ** 2))
        n = diff.size
        dy = (2.0 / n) * diff
        dW2 = np.outer(dy, h)
        db2 = dy.copy()
        dh = W2.T @ dy
        dz1 = dh * (z1 > 0).astype(np.float64)
        dW1 = np.outer(dz1, x)
        db1 = dz1.copy()
        return {
            'loss': round(loss, 5),
            'dW1': np.round(dW1, 5),
            'db1': np.round(db1, 5),
            'dW2': np.round(dW2, 5),
            'db2': np.round(db2, 5),
        }

    def forward_backward(self, x, W1, b1, W2, b2, y_true):
        return self.forward_and_backward(x, W1, b1, W2, b2, y_true)

    def backward(self, x, W1, b1, W2, b2, y_true):
        return self.forward_and_backward(x, W1, b1, W2, b2, y_true)