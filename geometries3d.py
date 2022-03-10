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

R = 1.5
r = 0.5
knots2d = np.array([ [[r,0,0],[R,0,0]] , [[r,r,0],[R,R,0]] , [[0,r,0],[0,R,0]] ], dtype = np.float64)
knots = np.concatenate(tuple([knots2d[:,:,None,:]]*2),2)
knots[...,1,2] = 0
knots[...,0,2] = 1

weights = np.ones(knots.shape[:3])
weights[1,:,:] = 1/np.sqrt(2)

basis1 = pinns.bspline.BSplineBasis(np.linspace(0,1,2),2)
basis2 = pinns.bspline.BSplineBasis(np.linspace(0,1,2),1)
basis3 = pinns.bspline.BSplineBasis(np.linspace(0,1,2),1)

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

pts , der = geom.importance_sampling(1000)

geoms = geom[:,0.5,:]
pts , der = geoms.importance_sampling(100000)

fig = plt.figure()
ax = plt.axes(projection ="3d")
ax.scatter(pts[:,0], pts[:,1], pts[:,2], s = 1)
print('Surface ',np.sum(np.sqrt(np.sum(der**2,-1))))

geoml = geoms[0.0:0.5,0.5]
pts , der = geoml.importance_sampling(100000)

fig = plt.figure()
ax = plt.axes(projection ="3d")
ax.scatter(pts[:,0], pts[:,1], pts[:,2], s = 1)
print('Line length ',np.sum(np.sqrt(np.sum(der**2,-1))))