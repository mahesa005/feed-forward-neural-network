import numpy as np


# all functions return np.ndarray; caller is responsible for wrapping in Tensor

def zero_init(shape):
    return np.zeros(shape)


def uniform_init(shape, lower=-0.1, upper=0.1, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(lower, upper, shape)


def normal_init(shape, mean=0.0, variance=1.0, seed=None):
    rng = np.random.default_rng(seed)
    return rng.normal(mean, np.sqrt(variance), shape)


def xavier_uniform_init(shape, seed=None):
    # scale = sqrt(6 / (fan_in + fan_out))
    fan_in, fan_out = shape[0], shape[1]
    scale = np.sqrt(6.0 / (fan_in + fan_out))
    rng   = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, shape)


def xavier_normal_init(shape, seed=None):
    # std = sqrt(2 / (fan_in + fan_out))
    fan_in, fan_out = shape[0], shape[1]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, std, shape)


def he_uniform_init(shape, seed=None):
    # scale = sqrt(6 / fan_in)
    fan_in = shape[0]
    scale  = np.sqrt(6.0 / fan_in)
    rng    = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, shape)


def he_normal_init(shape, seed=None):
    # std = sqrt(2 / fan_in)
    fan_in = shape[0]
    std    = np.sqrt(2.0 / fan_in)
    rng    = np.random.default_rng(seed)
    return rng.normal(0.0, std, shape)


INIT_MAP = {
    'zero':           zero_init,
    'uniform':        uniform_init,
    'normal':         normal_init,
    'xavier_uniform': xavier_uniform_init,
    'xavier_normal':  xavier_normal_init,
    'he_uniform':     he_uniform_init,
    'he_normal':      he_normal_init,
}


def get_initializer(name):
    if name not in INIT_MAP:
        raise ValueError(f"Unknown initializer: '{name}'. Choose from {list(INIT_MAP)}")
    return INIT_MAP[name]
