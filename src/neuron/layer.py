import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.engine.autodiff import Tensor
from src.neuron.base import Layer
from src.optim.initializers import get_initializer


class Dense(Layer):
    def __init__(self, in_features, out_features, use_bias=True,
                 initializer='xavier_normal', init_kwargs=None):
        """Initialize a fully connected (Dense) layer."""
        self.in_features  = in_features
        self.out_features = out_features
        self.use_bias     = use_bias

        init_fn = get_initializer(initializer)
        kwargs  = init_kwargs or {}

        W_data = init_fn((in_features, out_features), **kwargs)
        self.W = Tensor(W_data)

        if use_bias:
            self.b = Tensor(np.zeros(out_features))
        else:
            self.b = None

    def forward(self, x):
        """Forward pass: matrix multiplication and bias addition."""
        out = x @ self.W
        if self.b is not None:
            out = out + self.b  # broadcasts (out_features,) over batch dim
        return out

    def parameters(self):
        """Return weight and bias parameters."""
        params = [self.W]
        if self.b is not None:
            params.append(self.b)
        return params
