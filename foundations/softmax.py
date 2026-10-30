
import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:

        # Step 1: Subtract maximum value for numerical stability
        z = z - np.max(z)

        # Step 2: Calculate exponentials
        exp_z = np.exp(z)

        # Step 3: Divide each exponential by their sum
        result = exp_z / np.sum(exp_z)

        # Step 4: Round to 4 decimal places
        return np.round(result, 4)
