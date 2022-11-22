import numpy as np
import jax 
import jax.numpy as jnp
import pinns.bspline
import matplotlib.pyplot as plt
import unittest

import jax.config
jax.config.update("jax_enable_x64", True)


class TestBSpline(unittest.TestCase):

    def test_spline(self):
        
        basis = pinns.bspline.BSplineBasis(np.linspace(0,1,7),2)



if __name__ == '__main__':
    
    basis = pinns.bspline.BSplineBasis(np.linspace(0,1,7),2)
    
    x = np.linspace(0,1,1001)

    y = basis(x)
    plt.figure()
    plt.plot(x,y.T)

    y = basis(x, derivative = True)
    plt.figure()
    plt.plot(x,y.T)
    
    
    basis = pinns.bspline.BSplineBasis(np.array([0,0.1,0.5,0.5,0.9,1]),2)
    
    y = basis(x)
    plt.figure()
    plt.plot(x,y.T)

    y = basis(x, derivative = True)
    plt.figure()
    plt.plot(x,y.T)
    
   
    
    
    # JAX Bspl

    basis = pinns.bspline.BSplineBasisJAX(np.linspace(0,1,7),2)
    
    basisj = jax.jit(basis)
    
    x = np.linspace(0,1,1001)

    y = np.array(basisj(x))
    plt.figure()
    plt.plot(x,y.T)

     
    y = basis(x, derivative = True)
    plt.figure()
    plt.plot(x,y.T)
    
    
    basis = pinns.bspline.BSplineBasisJAX(np.array([0,0.1,0.5,0.5,0.9,1]),2)
    
    y = basis(x)
    print(y.shape)
    
    plt.figure()
    plt.plot(x,y.T)

    y = basis(x, derivative = True)
    plt.figure()
    plt.plot(x,y.T)
    
    plt.show()
