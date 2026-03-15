import numpy as np


class SGD:
    def __init__(self, parameters, learning_rate=0.01, l1_lambda=0.0, l2_lambda=0.0):
        """Initialize SGD optimizer with parameters and hyperparameters."""
        self.parameters = parameters
        self.lr         = learning_rate
        self.l1_lambda  = l1_lambda
        self.l2_lambda  = l2_lambda

    def step(self):
        for p in self.parameters:
            grad = p.grad.copy()
            if self.l2_lambda > 0:
                grad += self.l2_lambda * p.data        # L2: dL/dW += λW
            if self.l1_lambda > 0:
                grad += self.l1_lambda * np.sign(p.data)  # L1: dL/dW += λ·sign(W)
            p.data -= self.lr * grad

    def zero_grad(self):
        """Reset gradients to zero for all parameters."""
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)
