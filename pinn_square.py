import jax
import jax.numpy as jnp
import numpy as np
from jax.example_libraries import stax, optimizers
import matplotlib.pyplot as plt
import pinns 
import datetime
import jax.scipy.optimize
import jax.flatten_util

Phi_ref = lambda x:  np.exp(2*x[:,0])*np.sin(2*x[:,1])/8
Rhs = lambda x: 0
N_in = 5000
N_bd = 500
pts_inside = jnp.array(np.random.rand(N_in,2)*2-1)

tmp1 = np.random.randint(0, high=2, size=(N_bd,1))
tmp2 = np.random.randint(0, high=2, size=(N_bd,1))
xbd_train = (np.random.rand(N_bd,1)*2-1)*tmp1 + (1 - tmp1)*(tmp2*(-1)+(1-tmp2)*1)
tmp2 = np.random.randint(0, high=2, size=(N_bd,1))
ybd_train = (np.random.rand(N_bd,1)*2-1)*(1-tmp1) + tmp1*(tmp2*(-1)+(1-tmp2)*1)
pts_bd = jnp.array(np.concatenate((xbd_train[:],ybd_train[:]),1))
bd_vals = Phi_ref(pts_bd)

plt.figure()
plt.scatter(pts_bd[:,0],pts_bd[:,1],c='r',s=2)
plt.scatter(pts_inside[:,0],pts_inside[:,1],c='b',s=2)

nn_init, nn_apply = stax.serial(
    stax.Dense(22),
    stax.Sigmoid,
    stax.Dense(32),
    stax.Sigmoid,
    stax.Dense(32),
    stax.Sigmoid,
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
    lbd = jnp.sum((nn_apply(weights, pts_bd)[:,0] - bd_vals)**2)
    lpde = jnp.sum((pinns.operators.laplace(lambda x: nn_apply(weights, x))(pts_inside)[:,0] - Rhs(pts_inside))**2)
   
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
    return l,gr

print('Starting optimization')
# results = jax.scipy.optimize.minimize(loss_interface, x0 = weights_vector, method = 'bfgs', options = {'maxiter': 10})
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