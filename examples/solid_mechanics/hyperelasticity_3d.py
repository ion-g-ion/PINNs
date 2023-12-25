#%% Imports
import os 
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
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
config.update("jax_enable_x64", False)
rnd_key = jax.random.PRNGKey(1234)
np.random.seed(14124)

def create_geometry(key, scale = 1):
    R = 2
    r = 1
    Rp = 2
    d = 0.5
    
    knots = np.array( [ [[[r,0,0], [r,0,r], [0,0,r]], [[r,d,0], [r,d,r], [0,d,r]]], [[[R,0,0], [R,0,R], [0,0,R]], [[R,d,0], [R,d,R], [0,d,R]]] ])
    weights = np.ones(knots.shape[:3])
    weights[:,:,1] = 1/np.sqrt(2)
    
    basis1 = pinns.bspline.BSplineBasisJAX(np.array([-1,1]),1)
    basis2 = pinns.bspline.BSplineBasisJAX(np.array([-1,1]),1)
    basis3 = pinns.bspline.BSplineBasisJAX(np.array([-1,1]),2)

    geom1 = pinns.geometry.PatchNURBSParam([basis1, basis2, basis3], knots, weights, 0, 3, key)
    
    knots = np.array( [[[[0,0,r],[-(R-r),0,r]], [[0,d,r],[-(R-r),d,r]]], [[[0,0,R],[-(R-r),0,R]], [[0,d,R],[-(R-r),d,R]]]] )
    weights = np.ones(knots.shape[:3])
    
    basis1 = pinns.bspline.BSplineBasisJAX(np.array([-1,1]),1)
    basis2 = pinns.bspline.BSplineBasisJAX(np.array([-1,1]),1)
    basis3 = pinns.bspline.BSplineBasisJAX(np.array([-1,1]),1)

    geom2 = pinns.geometry.PatchNURBSParam([basis1, basis2, basis3], knots, weights, 0, 3, key)
    
    knots = np.array( [[[[r+Rp,-R,r], [-(R-r),-R,r], [-(R-r),0,r]], [[r+Rp,-r,r], [0,-r,r], [0,0,r]]], [[[r+Rp,-R,R], [-(R-r),-R,R], [-(R-r),0,R]], [[r+Rp,-r,R], [0,-r,R], [0,0,R]]]] )
    knots = np.transpose(knots,[0,2,1,3])
    knots = knots[:,:,::-1,:]
    weights = np.ones(knots.shape[:3])
    weights[:,1,:] = 1/np.sqrt(2)
    
    basis1 = pinns.bspline.BSplineBasisJAX(np.array([-1,1]),1)
    basis3 = pinns.bspline.BSplineBasisJAX(np.array([-1,1]),1)
    basis2 = pinns.bspline.BSplineBasisJAX(np.array([-1,1]),2)

    geom3 = pinns.geometry.PatchNURBSParam([basis1, basis2, basis3], knots, weights, 0, 3, key)
    
    
    knots = np.array( [[[[r+Rp,R+d,r], [-(R-r),R+d,r], [-(R-r),d,r]], [[r+Rp,r+d,r], [0,r+d,r], [0,d,r]]], [[[r+Rp,R+d,R], [-(R-r),R+d,R], [-(R-r),d,R]], [[r+Rp,r+d,R], [0,r+d,R], [0,d,R]]]] )
    knots = np.transpose(knots,[0,2,1,3])
    knots = knots[:,::-1,...]
    knots = knots[:,:,::-1,:]
    weights = np.ones(knots.shape[:3])
    weights[:,1,:] = 1/np.sqrt(2)
    
    
    basis1 = pinns.bspline.BSplineBasisJAX(np.array([-1,1]),1)
    basis2 = pinns.bspline.BSplineBasisJAX(np.array([-1,1]),2)
    basis3 = pinns.bspline.BSplineBasisJAX(np.array([-1,1]),1)

    geom4 = pinns.geometry.PatchNURBSParam([basis1, basis2, basis3], knots, weights, 0, 3, key)
    
    return  geom1, geom2, geom3, geom4

    
geom1, geom2, geom3, geom4 = create_geometry(rnd_key)
geoms = [geom1, geom2, geom3, geom4]



