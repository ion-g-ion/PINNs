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

R = 0.2
re = 0.1
knots = np.array([ [[1-R,0],[1,0]] , [[1-R,1-R-re],[1,1-re]] , [[1-R,1-R],[1,1]] , [[1-R-re,1-R],[1-re,1]] , [[-1+R+re,1-R],[-1+re,1]] , [[-1+R,1-R],[-1,1]] , [[-1+R,1-R-re],[-1,1-re]] , [[-1+R,-1+R+re],[-1,-1+re]] , [[-1+R,-1+R],[-1,-1]] , [[-1+R+re,-1+R],[-1+re,-1]] , [[1-R-re,-1+R],[1-re,-1]] , [[1-R,-1+R],[1,-1]] , [[1-R,-1+R+re],[1,-1+re]] , [[1-R,0],[1,0]] ])
weights = np.ones([knots.shape[0],knots.shape[1]])
weights[2,:] = 1/np.sqrt(2)
weights[5,:] = 1/np.sqrt(2)
weights[8,:] = 1/np.sqrt(2)
weights[11,:] = 1/np.sqrt(2)


basis1 = pinns.bspline.BSplineBasis(np.linspace(0,1,13),2)
basis2 = pinns.bspline.BSplineBasis(np.linspace(0,1,2),1)

geom = pinns.geometry.PatchNURBS([basis1, basis2], knots, weights, None)

plt.figure()
plt.scatter(knots[:,:,0].flatten(),knots[:,:,1].flatten())

x_in = geom.sample_inside(4000)
x_bd1,_ = geom.sample_boundary(1,0,250)
x_bd2,_ = geom.sample_boundary(1,1,250)

plt.figure()
plt.scatter(x_in[:,0],x_in[:,1],s=1,c='b')
plt.scatter(x_bd1[:,0],x_bd1[:,1],s=1,c='r')
plt.scatter(x_bd2[:,0],x_bd2[:,1],s=1,c='g')


nn_init, nn_apply = stax.serial(
    stax.Dense(15),
    stax.Tanh,
    stax.Dense(15),
    stax.Tanh,
    stax.Dense(15),
    stax.Tanh,
    stax.Dense(15),
    stax.Tanh,
    stax.Dense(1)
)


rng = jax.random.PRNGKey(123)

weights = nn_init(rng, (-1,2))

weights = weights[1] ## Weights are actually stored in second element of two value tuple

for w in weights:
    if w:
        w, b = w
        print("Weights : {}, Biases : {}".format(w.shape, b.shape))
        
@jax.jit
def loss(weights):
    lbd = (jnp.mean((nn_apply(weights, pts_bd)[:,0] - bd_vals)**2))
    lpde = (jnp.mean((pinns.operators.laplace(lambda x: nn_apply(weights, x))(pts_inside)[:] - Rhs(pts_inside)[:])**2))
    return lbd + 0.1*lpde

N_epochs = 100000

@jax.jit
def update(params, step_size = 0.001):

    grads = jax.grad(loss)(params)
    return [() if wb==() else (wb[0] - step_size * dwdb[0], wb[1] - step_size * dwdb[1]) for wb, dwdb in zip(params, grads)]
  
weights_vector, weights_unravel = jax.flatten_util.ravel_pytree(weights)

@jax.jit
def loss_handle(w):
    ws = weights_unravel(w)
    l = loss(ws)
    return l

@jax.jit
def lossgrad_handle(w):
    ws = weights_unravel(w)
    
    l = loss(ws)
    gr = jax.grad(loss)(ws)
    gr,_ = jax.flatten_util.ravel_pytree(gr)
    return l, gr

def loss_grad(w):
    l, gr = lossgrad_handle(jnp.array(w))
    return np.array(l.to_py()), np.array(gr.to_py()) 

