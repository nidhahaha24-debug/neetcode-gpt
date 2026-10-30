class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:

        if iterations == 0:
            return init

        x = float(init)

        for i in range(iterations):
            gradient = 2 * x
            x = x - learning_rate * gradient

        return round(x, 5)
