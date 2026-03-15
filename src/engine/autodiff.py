import numpy as np


def _sum_to_shape(grad, target_shape):
    """Reduce gradient to target_shape to reverse broadcasting."""
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, _children=(), requires_grad=True):
        """
        Wraps a numpy array with gradient tracking for automatic differentiation.

        Args:
            data          : array-like, converted to float64 numpy array
            _children     : tuple of Tensor nodes that produced this node
            requires_grad : if False, this node is skipped during backward
        """
        self.data          = np.array(data, dtype=float)
        self.grad          = np.zeros_like(self.data)
        self._backward     = lambda: None  # populated by each operation
        self._prev         = set(_children)
        self.requires_grad = requires_grad

    def __add__(self, other):
        """Element-wise addition with broadcasting support."""
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out   = Tensor(self.data + other.data, (self, other))
        def _backward():
            if self.requires_grad:
                self.grad  += _sum_to_shape(out.grad, self.shape)
            if other.requires_grad:
                other.grad += _sum_to_shape(out.grad, other.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        """Element-wise multiplication with broadcasting support."""
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out   = Tensor(self.data * other.data, (self, other))
        def _backward():
            if self.requires_grad:
                self.grad  += _sum_to_shape(out.grad * other.data, self.shape)
            if other.requires_grad:
                other.grad += _sum_to_shape(out.grad * self.data,  other.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        """Matrix multiplication. Supports batched inputs (B, in) @ (in, out) -> (B, out)."""
        out = Tensor(self.data @ other.data, (self, other))
        def _backward():
            if self.requires_grad:
                self.grad  += out.grad @ other.data.T  # (B, out) @ (out, in) -> (B, in)
            if other.requires_grad:
                other.grad += self.data.T @ out.grad   # (in, B) @ (B, out)  -> (in, out)
        out._backward = _backward
        return out

    def __pow__(self, exp):
        """Element-wise power: self ** exp, where exp is a scalar."""
        out = Tensor(self.data ** exp, (self,))
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * exp * (self.data ** (exp - 1))
        out._backward = _backward
        return out

    # derived ops — delegate to primitives above, no new _backward needed
    def __neg__(self):             return self * -1
    def __sub__(self, other):      return self + (-other)
    def __truediv__(self, other):  return self * other ** -1
    def __radd__(self, other):     return self + other
    def __rmul__(self, other):     return self * other
    def __rsub__(self, other):     return Tensor(other, requires_grad=False) + (-self)
    def __rtruediv__(self, other): return Tensor(other, requires_grad=False) * self ** -1

    @property
    def shape(self): return self.data.shape

    def sum(self, axis=None, keepdims=False):
        """Sum all elements or along an axis. Gradient broadcasts back to original shape."""
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))
        def _backward():
            if not self.requires_grad:
                return
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            self.grad += np.broadcast_to(grad, self.shape).copy()
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """Mean of elements or along an axis. Delegates to sum / n."""
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def exp(self):
        """Element-wise e^x. Backward: d/dx exp(x) = exp(x)."""
        out = Tensor(np.exp(self.data), (self,))
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * out.data
        out._backward = _backward
        return out

    def log(self):
        """Element-wise natural logarithm. Backward: d/dx log(x) = 1/x."""
        out = Tensor(np.log(self.data), (self,))
        def _backward():
            if self.requires_grad:
                self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def abs(self):
        """Element-wise absolute value. Backward: d/dx |x| = sign(x)."""
        out = Tensor(np.abs(self.data), (self,))
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * np.sign(self.data)
        out._backward = _backward
        return out

    def relu(self):
        """ReLU activation: max(0, x). Gradient is 1 where x > 0, else 0."""
        out = Tensor(np.maximum(0, self.data), (self,))
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0).astype(float)
        out._backward = _backward
        return out

    def sigmoid(self):
        """Sigmoid activation: 1 / (1 + e^-x). Backward: s * (1 - s)."""
        s   = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,))
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * out.data * (1.0 - out.data)
        out._backward = _backward
        return out

    def tanh(self):
        """Tanh activation. Backward: 1 - tanh(x)^2."""
        out = Tensor(np.tanh(self.data), (self,))
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (1.0 - out.data ** 2)
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        """
        Softmax activation along the given axis.
        Subtracts row-max before exp for numerical stability.
        Backward uses the efficient jacobian-vector product: s * (dout - sum(dout * s)).
        """
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        e       = np.exp(shifted)
        s       = e / e.sum(axis=axis, keepdims=True)
        out     = Tensor(s, (self,))
        def _backward():
            if self.requires_grad:
                self.grad += out.data * (out.grad - (out.grad * out.data).sum(axis=axis, keepdims=True))
        out._backward = _backward
        return out

    def elu(self, alpha=1.0):
        """
        ELU activation: x if x > 0 else alpha * (e^x - 1).
        Backward: 1 if x > 0 else output + alpha.
        """
        activated = np.where(self.data > 0, self.data, alpha * (np.exp(self.data) - 1.0))
        out = Tensor(activated, (self,))
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * np.where(self.data > 0, 1.0, out.data + alpha)
        out._backward = _backward
        return out

    def leaky_relu(self, alpha=0.01):
        """
        Leaky ReLU activation: x if x > 0 else alpha * x.
        Backward: 1 if x > 0 else alpha.
        """
        out = Tensor(np.where(self.data > 0, self.data, alpha * self.data), (self,))
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * np.where(self.data > 0, 1.0, alpha)
        out._backward = _backward
        return out

    def backward(self):
        """
        Compute gradients for all nodes in the computation graph via reverse-mode autodiff.
        Builds topological order with DFS, seeds this node's gradient to 1,
        then calls each node's _backward() in reverse order.
        """
        topo, visited = [], set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        """Reset this tensor's gradient to zero."""
        self.grad = np.zeros_like(self.data)

    def __repr__(self):
        return f"Tensor(shape={self.shape})"
