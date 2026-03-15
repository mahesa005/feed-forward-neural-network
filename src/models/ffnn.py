import numpy as np
import matplotlib.pyplot as plt

from src.engine.autodiff import Tensor
from src.neuron.layer import Dense
from src.neuron.activations import get_activation
from src.neuron.normalization import RMSNorm


class FFNN:
    def __init__(self, layer_sizes, activations, initializer='xavier_normal',
                 init_kwargs=None, use_bias=True, use_rmsnorm=False):
        """
        Args:
            layer_sizes  : list of ints, e.g. [11, 64, 32, 1]
            activations  : list of str, one per transition, e.g. ['relu', 'relu', 'sigmoid']
            initializer  : weight initializer name (see initializers.py)
            init_kwargs  : extra kwargs forwarded to the initializer function
            use_bias     : whether Dense layers include a bias term
            use_rmsnorm  : bool or list[bool] — whether to insert RMSNorm after each Dense
        """
        assert len(activations) == len(layer_sizes) - 1, \
            "len(activations) must equal len(layer_sizes) - 1"

        self.layer_sizes      = layer_sizes
        self.activation_names = activations
        self.layers           = []   # flat sequential list of all layers
        self.dense_layers     = []   # Dense layers only

        self._build(layer_sizes, activations, initializer, init_kwargs or {},
                    use_bias, use_rmsnorm)

    def _build(self, layer_sizes, activations, initializer, init_kwargs,
               use_bias, use_rmsnorm):
        """Build network layers with Dense and activation layers."""
        for i in range(len(layer_sizes) - 1):
            dense = Dense(layer_sizes[i], layer_sizes[i + 1],
                          use_bias=use_bias,
                          initializer=initializer,
                          init_kwargs=init_kwargs)
            self.dense_layers.append(dense)
            self.layers.append(dense)

            norm_flag = use_rmsnorm[i] if isinstance(use_rmsnorm, list) else use_rmsnorm
            if norm_flag:
                self.layers.append(RMSNorm(layer_sizes[i + 1]))

            self.layers.append(get_activation(activations[i]))

    def forward(self, x):
        """Run input through all layers sequentially. x: np.ndarray (batch_size, in_features). Returns Tensor."""
        t = x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)
        for layer in self.layers:
            t = layer(t)
        return t

    def __call__(self, x):
        return self.forward(x)

    def predict(self, x):
        """Returns raw numpy output (no gradient tracking)."""
        return self.forward(x).data

    def parameters(self):
        """Return all trainable Tensor parameters (W and b) from every layer."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def weight_parameters(self):
        """Only weight matrices W (no biases) — used for regularization."""
        return [dl.W for dl in self.dense_layers]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def plot_weight_distribution(self, layer_indices=None):
        """Plot histogram of weight values for each specified Dense layer."""
        indices = layer_indices if layer_indices is not None else range(len(self.dense_layers))
        fig, axes = plt.subplots(1, len(list(indices)), figsize=(4 * len(list(indices)), 3))
        if len(list(indices)) == 1:
            axes = [axes]
        for ax, idx in zip(axes, indices):
            w = self.dense_layers[idx].W.data.flatten()
            ax.hist(w, bins=50, color='steelblue', edgecolor='white')
            ax.set_title(f'Layer {idx} weights')
            ax.set_xlabel('value')
            ax.set_ylabel('count')
        plt.tight_layout()
        plt.show()

    def plot_gradient_distribution(self, layer_indices=None):
        """Histogram of weight gradients per Dense layer."""
        indices = layer_indices if layer_indices is not None else range(len(self.dense_layers))
        fig, axes = plt.subplots(1, len(list(indices)), figsize=(4 * len(list(indices)), 3))
        if len(list(indices)) == 1:
            axes = [axes]
        for ax, idx in zip(axes, indices):
            g = self.dense_layers[idx].W.grad.flatten()
            ax.hist(g, bins=50, color='tomato', edgecolor='white')
            ax.set_title(f'Layer {idx} gradients')
            ax.set_xlabel('value')
            ax.set_ylabel('count')
        plt.tight_layout()
        plt.show()

    def save(self, path):
        """Save model weights and config (layer_sizes, activations, W_i, b_i) to an .npz file."""
        state = {
            'layer_sizes':      np.array(self.layer_sizes),
            'activation_names': np.array(self.activation_names),
        }
        for i, dl in enumerate(self.dense_layers):
            state[f'W_{i}'] = dl.W.data
            if dl.b is not None:
                state[f'b_{i}'] = dl.b.data
        np.savez(path, **state)

    @classmethod
    def load(cls, path):
        """Reconstruct an FFNN from a saved .npz file."""
        data = np.load(path, allow_pickle=True)
        layer_sizes      = data['layer_sizes'].tolist()
        activation_names = data['activation_names'].tolist()

        model = cls(layer_sizes, activation_names, initializer='zero')

        for i, dl in enumerate(model.dense_layers):
            dl.W.data = data[f'W_{i}']
            if f'b_{i}' in data and dl.b is not None:
                dl.b.data = data[f'b_{i}']

        return model
