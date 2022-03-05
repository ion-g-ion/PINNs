import jax
import jax.numpy as jnp

def _aux_curl_2d(J,x):
    j = J(x)
    return (j[...,1,0]-j[...,0,1])[...,None]

def _aux_curl_3d(J,x):
    j = J(x)
    h1 = 1
    h2 = 1
    h3 = 1
    C1 = (j[...,2,1]-j[...,1,2])/(h2*h3)
    C2 = (j[...,0,2]-j[...,2,0])/(h2*h3)
    C3 = (j[...,1,0]-j[...,0,1])/(h1*h2)
    return jnp.concatenate((C1[...,None],C2[...,None],C3[...,None]), -1)

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
    return lambda x: _aux_curl_2d(J,x)

def curl3d(func):
    J = jax.vmap(jax.jacfwd(func))
    return lambda x: _aux_curl_3d(J,x)

def jacobian(func):
    J = jax.vmap(jax.jacfwd(func))
    return J

def jacobian_modified(func,Mat):
    J = jax.vmap(jax.jacfwd(func))
    return lambda x: (J(x)@Mat)[...,0,:]