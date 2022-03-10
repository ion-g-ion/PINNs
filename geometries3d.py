import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries import stax, optimizers
import matplotlib.pyplot as plt
import pinns 
import datetime
import jax.scipy.optimize
import jax.flatten_util
import scipy
import scipy.optimize

from jax.config import config
config.update("jax_enable_x64", True)
rnd_key = jax.random.PRNGKey(123)


knots2d = np.array([ [[-1,-1,0],[-1,0,0],[-1,1,0]] , [[0,-1,0],[0,0,0],[0,1,0]] , [[1,-1,0],[1,0,0],[1,1,0]] ], dtype = np.float64)
knots = np.concatenate(tuple([knots2d[None,...]]*5),0)
knots[0,...,2] = -2
knots[1,...,2] = -1
knots[2,...,2] = 0
knots[3,...,2] = 1
knots[4,...,2] = 2
weights = np.ones(knots.shape[:3])

weights[0,:,:] = 1/14
weights[0,:,:] = 1/14
weights[1,:,-1] = 1/12
weights[1,:,0] = 1/12

basis1 = pinns.bspline.BSplineBasis(np.linspace(0,1,4),2)
basis2 = pinns.bspline.BSplineBasis(np.linspace(0,1,3),1)
basis3 = pinns.bspline.BSplineBasis(np.linspace(0,1,3),1)

print(basis1)
print(basis2)
print(basis3)

geom = pinns.geometry.PatchNURBS([basis1,basis2,basis3],knots,weights, rnd_key)

fig = plt.figure()
ax = plt.axes(projection ="3d")
ax.scatter(knots[...,0].flatten(), knots[...,1].flatten(), knots[...,2].flatten(), s = 2)

Ps = geom.sample_inside(50000)

fig = plt.figure()
ax = plt.axes(projection ="3d")
ax.scatter(Ps[:,0], Ps[:,1], Ps[:,2], s = 1)