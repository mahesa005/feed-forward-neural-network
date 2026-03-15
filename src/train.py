import numpy as np

from src.engine.autodiff import Tensor

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def train(model, X_train, y_train, optimizer, loss_fn,
          epochs=100, batch_size=32,
          X_val=None, y_val=None,
          verbose=1):
    """
    Train model for a fixed number of epochs with mini-batch gradient descent.

    Args:
        model      : FFNN instance
        X_train    : np.ndarray, shape (N, features)
        y_train    : np.ndarray, shape (N,) or (N, outputs)
        optimizer  : SGD or Adam instance (already initialized with model.parameters())
        loss_fn    : callable(y_pred, y_true) -> scalar Tensor
        epochs     : number of full passes over the training data
        batch_size : number of samples per gradient update
        X_val      : optional validation inputs
        y_val      : optional validation targets
        verbose    : 0 = silent, 1 = tqdm progress bar per epoch

    Returns:
        history: dict with keys 'train_loss' and 'val_loss' (list per epoch)
    """
    history = {'train_loss': [], 'val_loss': []}
    n       = len(X_train)

    epoch_iter = range(epochs)
    if verbose == 1 and _HAS_TQDM:
        epoch_iter = tqdm(epoch_iter, desc='Training', unit='epoch')

    for epoch in epoch_iter:
        # shuffle training data each epoch
        idx      = np.random.permutation(n)
        X_shuf   = X_train[idx]
        y_shuf   = y_train[idx]

        batch_losses = []

        for start in range(0, n, batch_size):
            X_batch = X_shuf[start:start + batch_size]
            y_batch = y_shuf[start:start + batch_size]

            optimizer.zero_grad()

            y_pred = model.forward(X_batch)
            loss   = loss_fn(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.data))

        avg_train_loss = float(np.mean(batch_losses))
        history['train_loss'].append(avg_train_loss)

        if X_val is not None and y_val is not None:
            y_val_pred = model.forward(X_val)
            val_loss   = loss_fn(y_val_pred, y_val)
            history['val_loss'].append(float(val_loss.data))

        if verbose == 1 and _HAS_TQDM:
            desc = f'loss={avg_train_loss:.4f}'
            if history['val_loss']:
                desc += f'  val_loss={history["val_loss"][-1]:.4f}'
            epoch_iter.set_postfix_str(desc)
        elif verbose == 1 and not _HAS_TQDM:
            val_str = f'  val_loss={history["val_loss"][-1]:.4f}' if history['val_loss'] else ''
            print(f'Epoch {epoch + 1}/{epochs}  loss={avg_train_loss:.4f}{val_str}')

    return history
