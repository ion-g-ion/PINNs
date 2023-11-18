import numpy as np
import jax.numpy as jnp 
import jax
from jax.example_libraries import stax, optimizers
import pinns 
import pytest  

def test_dirichlet():
    """
    Test applying the Dirichlet 0 mask
    """
    init, nn =  pinns.pinns.DirichletMask(2, [(0,2), (0,1), (0,1)]*3, [{'dim': 0, 'end': -1}, {'dim': 1, 'end': 0}])
    ws = init(jax.random.PRNGKey(1234), (3,))[1]
    
    grid = np.meshgrid(np.linspace(0,2, 20), np.linspace(0,1, 20), np.linspace(0,1, 20), indexing = 'ij')
    inputs = np.array([list(i.flatten()) for i in grid]).T
    eval = nn(ws, inputs).reshape(grid[0].shape+(2,))
    assert np.linalg.norm(eval[-1,:,:,:]) < 1e-10
    assert np.linalg.norm(eval[:,0,:,:]) < 1e-10
    assert np.linalg.norm(eval) > 1e-10
    

