import jax
import jax.numpy as jnp

def gradient(func):
    J = jax.vmap(jax.jacfwd(func))
    return lambda x: J(x)[...,0,:]
    
def divergence(func):
    J = jax.vmap(jax.jacfwd(func))
    return lambda x: jnp.sum(jnp.diagonal(J(x),axis1=1,axis2=2),1, keepdims=True)

def laplace(func):
    H = jax.vmap(jax.hessian(func))
    return lambda x: jnp.sum(jnp.diagonal(H(x),axis1=1,axis2=2),1, keepdims=True)
