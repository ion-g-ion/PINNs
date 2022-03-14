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
re = 0.32
h = R
flux = 1.0
knots2d = np.array([ [[1-R,0],[1,0]] , [[1-R,1-R-re],[1,1-re]] , [[1-R,1-R],[1,1]] , [[1-R-re,1-R],[1-re,1]] , [[-1+R+re,1-R],[-1+re,1]] , [[-1+R,1-R],[-1,1]] , [[-1+R,1-R-re],[-1,1-re]] , [[-1+R,-1+R+re],[-1,-1+re]] , [[-1+R,-1+R],[-1,-1]] , [[-1+R+re,-1+R],[-1+re,-1]] , [[1-R-re,-1+R],[1-re,-1]] , [[1-R,-1+R],[1,-1]] , [[1-R,-1+R+re],[1,-1+re]] , [[1-R,0],[1,0]] ])
knots2d = np.pad(knots2d,((0,0),(0,0),(0,1)))
knots = np.concatenate((knots2d[None,...],knots2d[None,...]),0)
knots[0,:,:,2] = h/2
knots[1,:,:,2] = -h/2
knots = knots[:,:,::-1,:]

weights2d = np.ones([knots2d.shape[0],knots2d.shape[1]])
weights2d[2,:] = 1/np.sqrt(2)
weights2d[5,:] = 1/np.sqrt(2)
weights2d[8,:] = 1/np.sqrt(2)
weights2d[11,:] = 1/np.sqrt(2)
weights = np.concatenate((weights2d[None,...],weights2d[None,...]),0)

basis2 = pinns.bspline.BSplineBasis(np.linspace(0,1,13),2)
basis1 = pinns.bspline.BSplineBasis(np.linspace(0,1,2),1)
basis3 = pinns.bspline.BSplineBasis(np.linspace(0,1,2),1)

geom = pinns.geometry.PatchNURBS([basis1, basis2, basis3], knots, weights, None)


fig = plt.figure()
ax = plt.axes(projection ="3d")
ax.scatter3D(knots[...,0].flatten(), knots[...,1].flatten(), knots[...,2].flatten())
 
x_in,w_in = geom.importance_sampling(12000)
geom1 = geom[0.0,:,:]
geom2 = geom[1.0,:,:]
geom3 = geom[:,:,0.0]
geom4 = geom[:,:,1.0]
x_bd1, x_bdn1 = geom1.importance_sampling(9000)
x_bd2, x_bdn2 = geom2.importance_sampling(9000)
x_bd3, x_bdn3 = geom3.importance_sampling(9000)
x_bd4, x_bdn4 = geom4.importance_sampling(9000)
surface = lambda vs: np.sum(np.sqrt(np.sum(vs**2,-1)))
x_bdn1 /=surface(x_bdn1)
x_bdn2 /=surface(x_bdn2)
x_bdn3 /=surface(x_bdn3)*10
x_bdn4 /=surface(x_bdn4)
x_bd = np.concatenate((x_bd1, x_bd2, x_bd3, x_bd4), 0)

x_bdn = np.concatenate((x_bdn1, x_bdn2, x_bdn3, x_bdn4), 0)
# x_bdn = pinns.geometry.normalize(x_bdn)

xflux, vflux = geom[0.5,:,0.5].importance_sampling(10000)


geom1 = geom[0.1,:,1.0]
fig = plt.figure()
ax = plt.axes(projection ="3d")
# ax.scatter3D(x_in[...,0].flatten(), x_in[...,1].flatten(), x_in[...,2].flatten(),s=2)
ax.scatter3D(x_bd1[...,0].flatten(), x_bd1[...,1].flatten(), x_bd1[...,2].flatten(),s=2, c='r')
ax.scatter3D(x_bd2[...,0].flatten(), x_bd2[...,1].flatten(), x_bd2[...,2].flatten(),s=2, c='g')
ax.scatter3D(x_bd3[...,0].flatten(), x_bd3[...,1].flatten(), x_bd3[...,2].flatten(),s=2, c='b')
ax.scatter3D(x_bd4[...,0].flatten(), x_bd4[...,1].flatten(), x_bd4[...,2].flatten(),s=2, c='y')
ax.scatter3D(xflux[:,0],xflux[:,1],xflux[:,2], s= 2, c = 'k')
ax.quiver(x_bd[...,0].flatten(), x_bd[...,1].flatten(), x_bd[...,2].flatten(),x_bdn[...,0].flatten(), x_bdn[...,1].flatten(), x_bdn[...,2].flatten(), length = 0.2, normalize = True)
ax.view_init(90,0)
plt.savefig('3dplot.jpg')



xint, wint = geom.importance_sampling_3d(40000)

class Model(pinns.PINN):
    def __init__(self, rand_key, points):
        super().__init__()
        self.key = rand_key
        self.points = points
        nl = 32
        activation = stax.Tanh #  stax.elementwise(lambda x: jnp.tanh(x)*x)
        block = stax.serial(stax.FanOut(2),stax.parallel(stax.serial(stax.Dense(nl), activation, stax.Dense(nl), activation),stax.Dense(nl)),stax.FanInSum)
        #self.add_neural_network('Az', stax.serial(stax.Dense(10), stax.Tanh, stax.Dense(10), stax.Tanh, stax.Dense(10), stax.Tanh, stax.Dense(10), stax.Tanh, stax.Dense(1)), (-1,2))
        #self.add_neural_network('H', stax.serial(stax.Dense(10), stax.Tanh, stax.Dense(10), stax.Tanh, stax.Dense(10), stax.Tanh, stax.Dense(10), stax.Tanh, stax.Dense(2)), (-1,2))
        self.add_neural_network('B',stax.serial(block,block,block,block,block,stax.Dense(nl), activation,stax.Dense(3)),(-1,3))
        #self.add_neural_network('H',stax.serial(stax.Dense(nl), stax.Tanh,block,block,block,block,block,block,stax.Dense(nl), stax.Tanh,stax.Dense(nl), stax.Tanh, stax.Dense(nl), stax.Tanh,stax.Dense(3)),(-1,3))
        
    def loss(self, ws):
        
        Bbd = jnp.sum(self.neural_networks['B'](ws['B'],self.points['pts_bd'])*self.points['normals'], -1)
        # Hbd = jnp.sum(self.neural_networks['H'](ws['H'],self.points['pts_bd'])*self.points['normals'], -1)
        lbd_B = jnp.sum(Bbd**2)
        # lbd_H = jnp.mean(Hbd**2)
        
        mu = 1
        # Mat = jnp.array([[0.0,1.0],[-1.0,0.0]])
        # B = self.neural_networks['B'](ws['B'],self.points['pts_in'])
        # H = self.neural_networks['H'](ws['H'],self.points['pts_in'])
        # lmaterial = jnp.mean((B-mu*H)**2)
        
        lpde1 = jnp.mean((pinns.operators.curl3d(lambda x: (1/mu)*self.neural_networks['B'](ws['B'],x))(model.points['pts_in']))**2)
        lpde2 = jnp.mean((pinns.operators.divergence(lambda x: self.neural_networks['B'](ws['B'],x))(model.points['pts_in']))**2)
        
        lflux = (jnp.sum(self.neural_networks['B'](ws['B'],self.points['pts_flux'])*self.points['nflux'])-1)**2
        return 1*(lpde1+lpde2)+1000*lbd_B+10*lflux 

rnd_key = jax.random.PRNGKey(123)

model = Model(rnd_key, {'pts_in' : jnp.array(x_in), 'pts_bd' : jnp.array(x_bd), 'normals' : jnp.array(x_bdn), 'pts_flux' : jnp.array(xflux), 'nflux' : jnp.array(vflux)})

w0 = model.init_unravel()

loss_compiled = jax.jit(model.loss_handle)
lossgrad_compiled = jax.jit(model.lossgrad_handle)

def loss_grad(w):
    l, gr = lossgrad_compiled(jnp.array(w))
    return np.array(l.to_py()), np.array(gr.to_py()) 

print('Starting optimization')
tme = datetime.datetime.now()
#results = jax.scipy.optimize.minimize(loss_grad, x0 = weights_vector, method = 'bfgs', options = {'maxiter': 10})
#result = scipy.optimize.minimize(loss_grad, x0 = w0.to_py(), method = 'BFGS', jac = True, tol = 1e-8, options = {'disp' : True, 'maxiter' : 4000}, callback = lambda x: print(loss_compiled(x)))
result = scipy.optimize.minimize(loss_grad, x0 = w0.to_py(), method = 'L-BFGS-B', jac = True, tol = 1e-9, options = {'disp' : True, 'maxiter' : 8000, 'iprint': 1})
weights = model.weights_unravel(jnp.array(result.x))
tme = datetime.datetime.now() - tme

# opt_init, opt_update, get_params = optimizers.sgd(0.0025)
# opt_state = opt_init(model.weights)
# 
# loss = jax.jit(model.loss)
# for i in range(10000):
#     value, grads = jax.value_and_grad(loss)(get_params(opt_state))
#     opt_state = opt_update(i, grads, opt_state)
#     print(i,value)
print()
print('Elapsed time', tme)

x,y,z = np.meshgrid(np.linspace(0,1,5),np.linspace(0,1,40),np.linspace(0,1,4))
xyz = geom(np.concatenate((x.flatten()[:,None],y.flatten()[:,None],z.flatten()[:,None]),1))

B = model.neural_networks['B'](weights['B'], xyz)

fig = plt.figure()
ax = plt.axes(projection ="3d")
ax.quiver(xyz[:,0].flatten(), xyz[:,1].flatten(), xyz[:,2].flatten(),B[:,0].flatten(), B[:,1].flatten(), B[:,2].flatten(),colors = plt.cm.jet(np.sqrt(np.sum(B**2,-1))),normalize = True, length = 0.1)
plt.savefig('3dplot.jpg')

x,y,z = np.meshgrid(np.array([0.5]),np.linspace(0,1,80),np.linspace(0,1,8))
xyz = geom(np.concatenate((x.flatten()[:,None],y.flatten()[:,None],z.flatten()[:,None]),1))

B = model.neural_networks['B'](weights['B'], xyz)

fig = plt.figure()
plt.quiver(xyz[:,0].flatten(), xyz[:,1].flatten(),B[:,0].flatten(), B[:,1].flatten(),(np.sqrt(np.sum(B**2,-1))))

