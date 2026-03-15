from src.neuron.base import Layer


class Linear(Layer):
    """Identity activation — passes input through unchanged."""
    def forward(self, x):
        """Return input unchanged."""
        return x


class ReLU(Layer):
    """Rectified Linear Unit activation."""
    def forward(self, x):
        """Apply ReLU activation."""
        return x.relu()


class Sigmoid(Layer):
    """Sigmoid activation."""
    def forward(self, x):
        """Apply sigmoid activation."""
        return x.sigmoid()


class Tanh(Layer):
    """Hyperbolic tangent activation."""
    def forward(self, x):
        """Apply tanh activation."""
        return x.tanh()


class Softmax(Layer):
    """Softmax activation."""
    def __init__(self, axis=-1):
        """Initialize softmax with axis parameter."""
        self.axis = axis

    def forward(self, x):
        """Apply softmax activation."""
        return x.softmax(axis=self.axis)


class ELU(Layer):
    """Exponential Linear Unit activation."""
    def __init__(self, alpha=1.0):
        """Initialize ELU with alpha parameter."""
        self.alpha = alpha

    def forward(self, x):
        return x.elu(self.alpha)


class LeakyReLU(Layer):
    """Leaky Rectified Linear Unit activation."""
    def __init__(self, alpha=0.01):
        """Initialize LeakyReLU with alpha parameter."""
        self.alpha = alpha

    def forward(self, x):
        return x.leaky_relu(self.alpha)


_ACTIVATION_MAP = {
    'linear':     Linear,
    'relu':       ReLU,
    'sigmoid':    Sigmoid,
    'tanh':       Tanh,
    'softmax':    Softmax,
    'elu':        ELU,
    'leaky_relu': LeakyReLU,
}


def get_activation(name, **kwargs):
    """Get activation layer instance by name."""
    key = name.lower()
    if key not in _ACTIVATION_MAP:
        raise ValueError(f"Unknown activation: '{name}'. Choose from {list(_ACTIVATION_MAP)}")
    return _ACTIVATION_MAP[key](**kwargs)
