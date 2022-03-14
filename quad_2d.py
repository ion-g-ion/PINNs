from base64 import b16decode
from sunau import AUDIO_FILE_ENCODING_ADPCM_G721
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

def create_geometry(key, scale = 1):
    scale = scale
    Nt = 24                                                                
    lz = 40e-3                                                             
    Do = 72e-3                                                            
    Di = 51e-3                                                            
    hi = 13e-3                                                             
    bli = 3e-3                                                             
    Dc = 3.27640e-2                                                           
    hc = 7.55176e-3                                                           
    ri = 20e-3                                                           
    ra = 18e-3                                                           
    blc = hi-hc                                                           
    rm = (Dc*Dc+hc*hc-ri*ri)/(Dc*np.sqrt(2)+hc*np.sqrt(2)-2*ri)                 
    R = rm-ri
    O = np.array([rm/np.sqrt(2),rm/np.sqrt(2)])
    alpha1 = -np.pi*3/4       
    alpha2 = np.math.asin((hc-rm/np.sqrt(2))/R)
    alpha = np.abs(alpha2-alpha1)
    
    A = np.array([[O[0] - ri/np.sqrt(2), O[1] - ri/np.sqrt(2)], [O[0] - Dc, O[1] - hc]])
    b = np.array([[A[0,0]*ri/np.sqrt(2)+A[0,1]*ri/np.sqrt(2)],[A[1,0]*Dc+A[1,1]*hc]])
    C = np.linalg.solve(A,b)
    
    knots1 = np.array([[Do,0],[Do,Do * np.tan(np.pi/8)],[Do/np.sqrt(2),Do/np.sqrt(2)],[rm/np.sqrt(2),rm/np.sqrt(2)],[ri/np.sqrt(2),ri/np.sqrt(2)]])
    #knots2 = np.array([[Dc,hc],[Dc+blc,hi],[Di-bli,hi],[Di,hi-bli],[Di,0]])
    knots2 = np.array([[Di,0],[Di,hi-bli],[Di-bli,hi],[Dc+blc,hi],[Dc,hc]])
    knots3 = (knots1+knots2)/2
    knots3[-1,:] = C.flatten()
    knots = np.concatenate((knots1[None,...],knots3[None,...],knots2[None,...]),0)
    weights = np.ones(knots.shape[:2])
    weights[1,-1] = np.sin((np.pi-alpha)/2)
    basis2 = pinns.bspline.BSplineBasis(np.linspace(0,1,5),1)
    basis1 = pinns.bspline.BSplineBasis(np.array([0,1]),2)

    geom1 = pinns.geometry.PatchNURBS([basis1, basis2], knots, weights, key)
   
    knots2 = np.array([ [ [Dc,0],[Dc+blc,0],[Di-bli,0],[Di,0] ] , [[Dc,hc],[Dc+blc,hi],[Di-bli,hi],[Di,hi-bli]] ]) 
    weights = np.ones(knots2.shape[:2])
    
    basis1 = pinns.bspline.BSplineBasis(np.linspace(0,1,2),1)
    basis2 = pinns.bspline.BSplineBasis(np.array([0,0.2,0.8,1]),1)

    geom2 = pinns.geometry.PatchNURBS([basis1, basis2], knots2, weights, key)
   
    knots = np.array([ [ [0,0] , [Dc/2,0] , [Dc,0] ] , [ [ri/np.sqrt(2),ri/np.sqrt(2)] , [C[0,0],C[1,0]] , [Dc,hc] ]])
    
    basis1 = pinns.bspline.BSplineBasis(np.linspace(0,1,2),1)
    basis2 = pinns.bspline.BSplineBasis(np.array([0,1]),2)
    
    weights = np.ones(knots.shape[:2])
    weights[1,1]
    geom3 = pinns.geometry.PatchNURBS([basis1, basis2], knots, weights, key)
     
    return  geom1, geom2, geom3 

geom1, geom2, geom3 = create_geometry(rnd_key)

pts = geom2.sample_inside(1000)

plt.figure()
plt.scatter(pts[:,0], pts[:,1], s = 1)

pts = geom1.sample_inside(1000)
plt.scatter(pts[:,0],pts[:,1], s = 1)
plt.show()

pts = geom3.sample_inside(1000)
plt.scatter(pts[:,0],pts[:,1], s = 1)
plt.show()
