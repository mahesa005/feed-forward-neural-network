from abc import ABC, abstractmethod


class Layer(ABC):
    @abstractmethod
    def forward(self, x):
        """Forward pass through the layer."""

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        """Returns list of all Tensor parameters owned by this layer."""
        return []
