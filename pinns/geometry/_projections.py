import numpy as np
import jax 
import jax.numpy as jnp

def gap_to_convex_polytope(A: jax.Array, b: jax.Array, pts: jax.Array) -> jax.Array:
    """
    Compute the distance from the given points to the boundary of the polytope Ax<=b.
    If the points are outside, the distance is 0. 
    

    Args:
        A (jax.Array): the matrix A.
        b (jax.Array): the vector b.
        pts (jax.Array): the poitns.

    Returns:
        jax.Array: the distance.
    """
    
    res = -jnp.einsum('mn,kn->km', A, pts) + b # distance from boundry. Positive if inside negative if outside
    
    dist = jnp.min(jax.numpy.where(res<=0.0, 0.0, res), axis=1)

    return  dist
    
    
    
    
    
    
    
    
    
