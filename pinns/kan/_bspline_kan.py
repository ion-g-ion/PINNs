import jax 
import jax.numpy as jnp 
import stax
from typing import Tuple, Callable

def get_spline_basis(x, knots, deg):
    grid = knots
    x = jnp.expand_dims(x, axis=1)
    basis_splines = ((x >= knots[:, :-1]) & (x < knots[:, 1:])).astype(float)
    for k in range(1, deg + 1):
        left_term = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)])
        right_term = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)])

        basis_splines = left_term * basis_splines[:, :-1] + right_term * basis_splines[:, 1:]

    return basis_splines
    
def KANLayer(output_size: int, degree: int, activation: Callable, knots: jnp.array) -> Tuple[Callable, Callable]:

    def init_fun(rng, input_shape):
        
        return nn_tuple[0](rng, input_shape)


    def apply_fun(params, inputs, **kwargs):
        
    
    return init_fun, apply_fun