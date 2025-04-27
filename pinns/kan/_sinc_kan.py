import falx.nnx as nnx
import jax.numpy as jnp
from jax import random
from jax.nn import tanh


class KAN(nnx.Module):
    layers: nnx.Node
    activation: nnx.Node
    normalizer: nnx.Node

    def __init__(self, features, interval, normalizer, key, degree=10, activation='tanh'):
        keys = random.split(key, len(features) + 1)
        self.layers = nnx.List(
            [
                SincLayers(f_in, f_out, degree, interval, key)
                for f_in, f_out, key in zip(features[:-1], features[1:], keys)
            ]
        )
        self.activation = tanh if activation == 'tanh' else None
        self.normalizer = nnx.Param([normalizer])

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SincLayers(nnx.Module):
    degree: int
    len_h: int
    init_h: float
    coeffs: nnx.Param
    decay: str
    beta: nnx.Param
    alpha: nnx.Param
    activation: nnx.Node
    skip: bool
    frozen_k: jnp.array
    frozen_h: jnp.array

    def __init__(self, input_dim, output_dim, degree, key, init_h, len_h=2, activation='tanh', decay='inverse', skip=True):
        self.degree = degree
        self.len_h = len_h
        self.init_h = init_h
        self.decay = decay
        self.skip = skip

        # Initialize parameters
        coeffs_init = random.normal(key, (input_dim, output_dim, len_h, degree + 1)) / jnp.sqrt(
            input_dim * (degree + 1)
        )
        self.coeffs = nnx.Param(coeffs_init)
        self.alpha = nnx.Param(jnp.ones((input_dim, output_dim)))
        self.beta = nnx.Param(jnp.zeros((output_dim,)))
        self.activation = tanh if activation == 'tanh' else None

        # Generate frozen parameters during initialization
        self.frozen_k = jnp.expand_dims(
            jnp.arange(-jnp.floor(self.degree / 2), jnp.ceil(self.degree / 2) + 1), axis=(0, 1)
        )
        if self.decay == 'inverse':
            self.frozen_h = 1 / (self.init_h * (1 + jnp.arange(self.len_h)))
        elif self.decay == 'exp':
            self.frozen_h = 1 / (self.init_h ** (1 + jnp.arange(self.len_h)))
        else:
            raise ValueError(f"{self.decay} does not exist")
        self.frozen_h = jnp.expand_dims(self.frozen_h, axis=(0, 2))

    def __call__(self, x):
        # Linear skip connection
        if self.skip:
            y_eqt = x @ self.alpha + self.beta

        # Apply activation
        if self.activation:
            x = self.activation(x)

        # Interpolation computation
        x = jnp.tile(jnp.expand_dims(x, axis=(1, 2)), (1, 1, self.degree + 1))
        x = x / self.frozen_h + self.frozen_k
        x_interp = jnp.sinc(x)

        # Apply coefficients
        y = jnp.einsum("ikd,iokd->o", x_interp, self.coeffs)

        # Add skip connection result
        if self.skip:
            y = y_eqt + y

        return y
