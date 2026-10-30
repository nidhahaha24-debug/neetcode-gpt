
import numpy as np
from numpy.typing import NDArray


class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:

        # Multiply feature matrix with weights
        prediction = np.dot(X, weights)

        # Round to 5 decimal places
        return np.round(prediction, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:

        # Calculate difference
        error = model_prediction - ground_truth

        # Square the errors and calculate mean
        mse = np.mean(error ** 2)

        return round(float(mse), 5)

