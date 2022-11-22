import jax 
import jax.numpy as jnp
import numpy as np
from jax.example_libraries import stax, optimizers


def func(x):
    # return (x[...,0,None]**3,2*jnp.sin(x[...,1,None]))
    # print(x.shape)
    return jnp.concatenate((x[...,0,None]**3,2*jnp.sin(x[...,1,None])),-1)

def func_scalar(x):
    return x[...,0,None]+2*x[...,1,None]

x = jnp.array(np.random.rand(400,2))

J = jax.vmap(jax.jacfwd(func))
D = divergence(func)

div_ref = x[...,0]**2 * 3 + 2*jnp.cos(x[...,1])
div = D(x)
gr = gradient(func_scalar)

nn_init, nn_apply = stax.serial(
    stax.Dense(2),
    stax.Sigmoid,
    stax.Dense(12),
    stax.Sigmoid,
    stax.Dense(12),
    stax.Sigmoid,
    stax.Dense(1)
)

rng = jax.random.PRNGKey(123)

weights = nn_init(rng, (-1,2))

weights = weights[1] ## Weights are actually stored in second element of two value tuple

for w in weights:
    if w:
        w, b = w
        print("Weights : {}, Biases : {}".format(w.shape, b.shape))
        
