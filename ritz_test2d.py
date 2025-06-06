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
key = jax.random.PRNGKey(0)

b1 = pinns.bspline.BSplineBasis(np.linspace(-1,1,4),2)
b2 = pinns.bspline.BSplineBasis(np.linspace(-1,1,2),1)

knots = np.concatenate( ( np.array([[0,1],[0,1],[0,1],[2,2],[3,3]])[...,None] , np.array([[0,0],[1,1],[3,2],[3,2],[3,2]])[...,None] ),-1)
weights = np.array([[1,1],[1,1],[1/np.sqrt(2),1/np.sqrt(2)],[1,1],[1,1]])
geom = pinns.geometry.PatchNURBS([b1, b2], knots, weights, key)

def GetPoints():
    x_in, ws_in = geom.quadrature(32)
    x_bd1,_ = geom[:,-1].importance_sampling(500)
    x_bd2,_ = geom[:,1].importance_sampling(500)
    x_bd3,_ = geom[-1,:].importance_sampling(500)
    x_bd4,_ = geom[1,:].importance_sampling(500)
    return {'pts_in' : x_in, 'ws_in' : ws_in, 'pts_bd' : np.concatenate((x_bd1,x_bd2,x_bd3,x_bd4),0) }

points = GetPoints()
plt.figure()
plt.scatter(points['pts_in'][:,0],points['pts_in'][:,1], s=1)
plt.scatter(points['pts_bd'][:,0],points['pts_bd'][:,1], s=1)

class Model(pinns.PINN):
    def __init__(self, rand_key):
        super().__init__()
        self.key = rand_key

        nl = 32
        
        block = stax.serial(stax.FanOut(2),stax.parallel(stax.serial(stax.Dense(nl), stax.Tanh, stax.Dense(nl), stax.Tanh),stax.Dense(nl)),stax.FanInSum)
    
        self.add_neural_network('u',stax.serial(block,block,block,block,block,block,block,block,stax.Tanh,stax.Dense(1)),(-1,2))


    def loss_pde(self, ws, points):
        grad = pinns.operators.gradient(lambda x : self.neural_networks['u'](ws['u'],x))(points['pts_in'])
        nu = 1/3
        fval = (lambda x : self.neural_networks['u'](ws['u'],x))(points['pts_in']) 
        
        lpde = 0.5*nu*jnp.dot(jnp.sum(grad**2,-1),points['ws_in']) -jnp.dot( fval.flatten() ,points['ws_in'])
        return lpde
    
    def loss_bd(self, ws, points):
        lbd_in = (jnp.mean((self.neural_networks['u'](ws['u'], points['pts_bd'])[:,0] - 0.0)**2))
        lbd_out = 0
        return lbd_in, lbd_out
    
    def loss(self, ws, points):
        lbd_in, lbd_out = self.loss_bd(ws, points)        
        lpde = self.loss_pde(ws, points)

        return 1*lpde+100*lbd_in+100*lbd_out

    def loss_handle(self, w, p):
        ws = self.weights_unravel(w)
        l = self.loss(ws, p)
        return l


    def lossgrad_handle(self, w, p):
        ws = self.weights_unravel(w)
            
        l = self.loss(ws, p )
        gr = jax.grad(self.loss, argnums=0)(ws,p)
            
        gr,_ = jax.flatten_util.ravel_pytree(gr)
        return l, gr

rnd_key = jax.random.PRNGKey(123)


model = Model(rnd_key)

w0 = model.init_unravel()
weights = w0


loss_compiled = jax.jit(model.loss_handle)
lossgrad_compiled = jax.jit(model.lossgrad_handle)

print('Starting optimization')

for i in range(1):
    print('Epoch %d'%(i+1))
    points = GetPoints()
    def loss_grad(w):
        l, gr = lossgrad_compiled(jnp.array(w), points)
        return np.array(l.to_py()), np.array(gr.to_py()) 

    tme = datetime.datetime.now()
    #results = jax.scipy.optimize.minimize(loss_grad, x0 = weights_vector, method = 'bfgs', options = {'maxiter': 10})
    # result = scipy.optimize.minimize(loss_grad, x0 = w0.to_py(), method = 'BFGS', jac = True, tol = 1e-8, options = {'disp' : True, 'maxiter' : 400}, callback = lambda x: print(loss_compiled(x)))
    result = scipy.optimize.minimize(loss_grad, x0 = weights.to_py(), method = 'L-BFGS-B', jac = True, tol = 1e-9, options = {'disp' : True, 'maxiter' : 10000, 'iprint': 1})
    weights = (jnp.array(result.x))
    tme = datetime.datetime.now() - tme

weights = model.weights_unravel(jnp.array(result.x))

print()
print('Elapsed time', tme)

x,y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
xy = geom(np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),1))

Az = model.neural_networks['u'](weights['u'], xy).reshape(x.shape)

plt.figure()
plt.contourf(xy[:,0].reshape(x.shape),xy[:,1].reshape(x.shape),Az, levels = 32 )
plt.colorbar()
