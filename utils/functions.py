# libraries
import numpy as np

class Functions():
    """Functions for DNN"""

    def __init__(self):
        return self

    # Activation Functions
    def step_function(self, x):
        y = x > 0
        return y.astype(np.int)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x):
        c = np.max(x)
        exp_x = np.exp(x - c) # avoid Overflow
        sum_exp_x = np.sum(exp_x)
        y = exp_x / sum_exp_x
        return y

    def relu(self, x):
        return np.maximum(0, x)

    def leaky_relu(self, x):
        a = 0.01
        if x < 0:
            return a * x
        else: # x >= 0
            return x

    # Loss Functions
    def mean_squared_error(self, y, t):
        return 0.5 * np.sum(np.power((y - t), 2))

    def root_mean_squared_error(self, y, t):
        return np.sqrt(self.mean_squared_error(y, t))

    def cross_entropy_error(self, y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y + delta)) # avoid negative infinite

    def corss_entropy_error_batch(self, y, t):
        """cross entropy error for batch and one-hot encoded labels"""
        delta = 1e-7
        if y.ndim == -1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        batch_size = y.shape[0]
        return -np.sum(t * np.log(y + delta)) / batch_size