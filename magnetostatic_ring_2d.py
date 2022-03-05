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

R = 0.5
re = 0.2
h = R*2
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

x_in = geom.sample_inside(6000)
x_bd1,_ = geom.sample_boundary(1,0,1000)
x_bd2,_ = geom.sample_boundary(1,1,1000)

plt.figure()
plt.scatter(x_in[:,0],x_in[:,1],s=1,c='b')
plt.scatter(x_bd1[:,0],x_bd1[:,1],s=1,c='r')
plt.scatter(x_bd2[:,0],x_bd2[:,1],s=1,c='g')

class Model(pinns.PINN):
    def __init__(self, rand_key, points):
        super().__init__()
        self.key = rand_key
        self.points = points
        nl = 32
        block = stax.serial(stax.FanOut(2),stax.parallel(stax.serial(stax.Dense(nl), stax.Tanh, stax.Dense(nl), stax.Tanh),stax.Dense(nl)),stax.FanInSum)
        #self.add_neural_network('Az', stax.serial(stax.Dense(10), stax.Tanh, stax.Dense(10), stax.Tanh, stax.Dense(10), stax.Tanh, stax.Dense(10), stax.Tanh, stax.Dense(1)), (-1,2))
        #self.add_neural_network('H', stax.serial(stax.Dense(10), stax.Tanh, stax.Dense(10), stax.Tanh, stax.Dense(10), stax.Tanh, stax.Dense(10), stax.Tanh, stax.Dense(2)), (-1,2))
        self.add_neural_network('Az',stax.serial(block,block,block,block,block,block,stax.Tanh,stax.Dense(1)),(-1,2))
        self.add_neural_network('H',stax.serial(block,block,block,block,block,block,stax.Tanh,stax.Dense(2)),(-1,2))
        
    def loss(self, ws):
        
        lbd_in = (jnp.mean((self.neural_networks['Az'](ws['Az'], model.points['pts_bd_in'])[:,0] - 0)**2))
        lbd_out = (jnp.mean((self.neural_networks['Az'](ws['Az'], model.points['pts_bd_out'])[:,0] - 1.0)**2))
        
        mu = 10
        Mat = jnp.array([[0.0,1.0],[-1.0,0.0]])
        B = pinns.operators.jacobian_modified(lambda x: self.neural_networks['Az'](ws['Az'],x),Mat)(model.points['pts_inside'])
        lmaterial = jnp.mean((B-mu*self.neural_networks['H'](ws['H'], model.points['pts_inside']))**2)
        
        lpde = jnp.mean((pinns.operators.curl2d(lambda x: self.neural_networks['H'](ws['H'],x))(model.points['pts_inside']))**2)
        
        return 0.1*lpde+lmaterial+10*lbd_in+10*lbd_out

rnd_key = jax.random.PRNGKey(123)

model = Model(rnd_key, {'pts_bd_in' : x_bd1, 'pts_bd_out' : x_bd2, 'pts_inside' : x_in})

w0 = model.init_unravel()

loss_compiled = jax.jit(model.loss_handle)
lossgrad_compiled = jax.jit(model.lossgrad_handle)

def loss_grad(w):
    l, gr = lossgrad_compiled(jnp.array(w))
    return np.array(l.to_py()), np.array(gr.to_py()) 

print('Starting optimization')
tme = datetime.datetime.now()
#results = jax.scipy.optimize.minimize(loss_grad, x0 = weights_vector, method = 'bfgs', options = {'maxiter': 10})
# result = scipy.optimize.minimize(loss_grad, x0 = w0.to_py(), method = 'BFGS', jac = True, tol = 1e-8, options = {'disp' : True, 'maxiter' : 400}, callback = lambda x: print(loss_compiled(x)))
result = scipy.optimize.minimize(loss_grad, x0 = w0.to_py(), method = 'L-BFGS-B', jac = True, tol = 1e-9, options = {'disp' : True, 'maxiter' : 2000, 'iprint': 1})
weights = model.weights_unravel(jnp.array(result.x))
tme = datetime.datetime.now() - tme

print()
print('Elapsed time', tme)

x,y = np.meshgrid(np.linspace(0,1,10),np.linspace(0,1,100))
xy = geom(np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),1))

Az = model.neural_networks['Az'](weights['Az'], xy).reshape(x.shape)

plt.figure()
plt.contourf(xy[:,0].reshape(x.shape),xy[:,1].reshape(x.shape),Az)