import jax
import jax.numpy as jnp
import numpy as np
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

Phi_ref = lambda x:  np.exp(0.5*x[:,0])*np.sin(0.5*x[:,1])/8
Rhs = lambda x: 0.0
N_in = 2500
N_bd = 500
pts_inside = jnp.array(np.random.rand(N_in,2)*2-1,  dtype = jnp.float64)

tmp1 = np.random.randint(0, high=2, size=(N_bd,1))
tmp2 = np.random.randint(0, high=2, size=(N_bd,1))
xbd_train = (np.random.rand(N_bd,1)*2-1)*tmp1 + (1 - tmp1)*(tmp2*(-1)+(1-tmp2)*1)
tmp2 = np.random.randint(0, high=2, size=(N_bd,1))
ybd_train = (np.random.rand(N_bd,1)*2-1)*(1-tmp1) + tmp1*(tmp2*(-1)+(1-tmp2)*1)
pts_bd = jnp.array(np.concatenate((xbd_train[:],ybd_train[:]),1), dtype = jnp.float64)
bd_vals = Phi_ref(pts_bd)

plt.figure()
plt.scatter(pts_bd[:,0],pts_bd[:,1],c='r',s=2)
plt.scatter(pts_inside[:,0],pts_inside[:,1],c='b',s=2)

nn_init, nn_apply = stax.serial(
    stax.Dense(15),
    stax.Tanh,
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
    lbd = jnp.sqrt(jnp.sum((nn_apply(weights, pts_bd)[:,0] - bd_vals)**2))
    lpde = jnp.sqrt(jnp.sum((pinns.operators.laplace(lambda x: nn_apply(weights, x))(pts_inside)[:,0] - Rhs(pts_inside))**2))
   
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
    return (l.to_py()), gr.to_py() 


print('Starting optimization')
# results = jax.scipy.optimize.minimize(loss_interface, x0 = weights_vector, method = 'bfgs', options = {'maxiter': 10})
result = scipy.optimize.minimize(loss_grad, x0 = weights_vector.to_py(), method = 'BFGS', jac = True, options = {'disp' : True, 'maxiter' : 10000}, callback = lambda x: print(loss_handle(x)))
# result = scipy.optimize.minimize(loss_grad, x0 = weights_vector.to_py(), method = 'L-BFGS-B', jac = True, options = {'disp' : True, 'maxiter' : 1500, 'iprint': 1})

print('Ready')
# for epoch in range(N_epochs):
#     tme = datetime.datetime.now()
#     weights = update(weights, 0.0005)
#     tme = datetime.datetime.now() - tme
#     print()
#     print('Iteration ',epoch+1,' time ',tme, ' loss ', loss(weights))
   
    
x,y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
xy = np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),1)

plt.figure()
plt.contourf(x,y,nn_apply(weights,xy).reshape(x.shape))
plt.colorbar()

plt.figure()
plt.contourf(x,y,Phi_ref(xy).reshape(x.shape))
plt.colorbar()

plt.figure()
plt.contourf(x,y,nn_apply(weights,xy).reshape(x.shape) - Phi_ref(xy).reshape(x.shape))
plt.colorbar()