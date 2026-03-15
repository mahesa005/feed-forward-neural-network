import numpy as np


class Adam:
    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, l1_lambda=0.0, l2_lambda=0.0):
        """Initialize Adam optimizer with parameters and hyperparameters."""
        self.parameters = parameters
        self.lr         = learning_rate
        self.beta1      = beta1
        self.beta2      = beta2
        self.eps        = eps
        self.l1_lambda  = l1_lambda
        self.l2_lambda  = l2_lambda
        self.t          = 0  # timestep counter

        # per-parameter running estimates of 1st and 2nd gradient moments
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        """Update parameters using bias-corrected first and second moments."""
        self.t += 1
        for i, p in enumerate(self.parameters):
            grad = p.grad.copy()
            if self.l2_lambda > 0:
                grad += self.l2_lambda * p.data
            if self.l1_lambda > 0:
                grad += self.l1_lambda * np.sign(p.data)

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad ** 2

            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)  # bias-corrected 1st moment
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)  # bias-corrected 2nd moment

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Reset gradients to zero for all parameters."""
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)
