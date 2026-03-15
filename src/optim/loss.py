import numpy as np

from src.engine.autodiff import Tensor

_EPS = 1e-12  # clip bound to prevent log(0)


def mse_loss(y_pred, y_true):
    """Mean squared error: mean((y_pred - y_true)^2)."""
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    diff = y_pred - y_true
    return (diff * diff).mean()


def bce_loss(y_pred, y_true):
    """Binary cross-entropy: -mean(y*log(p) + (1-y)*log(1-p))."""
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    # clip predictions to avoid log(0) or log(negative)
    p = Tensor(np.clip(y_pred.data, _EPS, 1.0 - _EPS), requires_grad=False)
    return -(y_true * p.log() + (1.0 - y_true) * (1.0 - p).log()).mean()


def cce_loss(y_pred, y_true):
    """Categorical cross-entropy: -mean(sum(y_true * log(y_pred), axis=-1))."""
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    # clip predictions
    p = Tensor(np.clip(y_pred.data, _EPS, 1.0), requires_grad=False)
    return -(y_true * p.log()).sum(axis=-1).mean()
