import jax
import jax.numpy as jnp

def _aux_curd_2d(J,x):
    j = J(x)
    return (j[...,1,0]-j[...,0,1])[...,None]

def gradient(func):
    J = jax.vmap(jax.jacfwd(func))
    return lambda x: J(x)[...,0,:]
    
def divergence(func):
    J = jax.vmap(jax.jacfwd(lambda x: func(x)))
    return lambda x: jnp.sum(jnp.diagonal(J(x), axis1 = 1, axis2=2), -1, keepdims=True)

def laplace(func):
    H = jax.vmap(jax.hessian(func))
    return lambda x: jnp.sum(jnp.diagonal(H(x)[:,0,:,:],axis1=1,axis2=2),1)[...,None]

def curl2d(func):
    J = jax.vmap(jax.jacfwd(func))
    return lambda x: _aux_curd_2d(J,x)
