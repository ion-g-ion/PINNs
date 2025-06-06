import numpy as np
import jax 
import jax.numpy as jnp
import pinns
import matplotlib.pyplot as plt
import unittest

import jax.config
jax.config.update("jax_enable_x64", True)





if __name__ == '__main__':
    
    basis = pinns.functions.BSplineBasisJAX(np.linspace(0,1,7),2)
    
    x = np.linspace(0,1,1001)

    # y = basis(x)
    # plt.figure()
    # plt.plot(x,y.T)

    # y = basis(x, derivative = True)
    # plt.figure()
    # plt.plot(x,y.T)
    # 
    # 
    # basis = pinns.functions.BSplineBasisJAX(np.array([0,0.1,0.5,0.5,0.9,1]),2)
    # 
    # y = basis(x)
    # plt.figure()
    # plt.plot(x,y.T)

    # y = basis(x, derivative = True)
    # plt.figure()
    # plt.plot(x,y.T)
    # 
    # 
    # 
    # 
    # # JAX Bspl

    # basis = pinns.functions.BSplineBasisJAX(np.linspace(0,1,7),2)
    # 
    # basisj = jax.jit(basis)
    # 
    # x = np.linspace(0,1,1001)

    # y = np.array(basisj(x))
    # plt.figure()
    # plt.plot(x,y.T)

    # 
    # y = basis(x, derivative = True)
    # plt.figure()
    # plt.plot(x,y.T)
    # 
    # 
    # basis = pinns.functions.BSplineBasisJAX(np.array([0,0.1,0.5,0.5,0.9,1]),2)
    # 
    # y = basis(x)
    # print(y.shape)
    # 
    # plt.figure()
    # plt.plot(x,y.T)

    # y = basis(x, derivative = True)
    # plt.figure()
    # plt.plot(x,y.T)
    # 
    # plt.show()


    basis = pinns.functions.PiecewiseBernsteinBasisJAX(np.array([0,0.1,0.3,0.7,0.9,1]),3)
    
    y = basis(x)
    print(y.shape)
    
    plt.figure()
    plt.title("Bernstein")
    plt.plot(x,y.T)

    y = basis(x, derivative = True)
    plt.figure()
    plt.plot(x,y.T)
    
    plt.show()