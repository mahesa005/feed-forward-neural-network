import numpy as np

from src.engine.autodiff import Tensor
from src.neuron.base import Layer


class RMSNorm(Layer):
    def __init__(self, features, eps=1e-8):
        """Initialize RMSNorm layer with learnable scale parameter."""
        self.features = features
        self.eps      = eps
        # learnable per-feature scale, initialized to ones
        self.gamma    = Tensor(np.ones(features))

    def forward(self, x):
        """Normalize input and apply learnable scale."""
        # rms = sqrt(mean(x^2, axis=-1) + eps), shape (B, 1)
        rms = ((x * x).mean(axis=-1, keepdims=True) + self.eps) ** 0.5
        return (x / rms) * self.gamma  # gamma broadcasts over batch dim

    def parameters(self):
        """Return learnable gamma scale parameter."""
        return [self.gamma]
