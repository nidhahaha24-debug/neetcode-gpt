import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        x_arr = np.array(x, dtype=np.float64)
        gamma_arr = np.array(gamma, dtype=np.float64)
        beta_arr = np.array(beta, dtype=np.float64)
        run_mean = np.array(running_mean, dtype=np.float64)
        run_var = np.array(running_var, dtype=np.float64)
        if training:
            batch_mean = np.mean(x_arr, axis=0)
            batch_var = np.var(x_arr, axis=0)
            x_hat = (x_arr - batch_mean) / np.sqrt(batch_var + eps)
            y = gamma_arr * x_hat + beta_arr
            new_mean = (1.0 - momentum) * run_mean + momentum * batch_mean
            new_var = (1.0 - momentum) * run_var + momentum * batch_var
        else:
            x_hat = (x_arr - run_mean) / np.sqrt(run_var + eps)
            y = gamma_arr * x_hat + beta_arr
            new_mean = run_mean
            new_var = run_var
        y_list = np.round(y, 4).tolist()
        mean_list = np.round(new_mean, 4).tolist()
        var_list = np.round(new_var, 4).tolist()
        return y_list, mean_list, var_list