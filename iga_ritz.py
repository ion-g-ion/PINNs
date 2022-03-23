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

b1 = pinns.bspline.BSplineBasis(np.linspace(0,1,4),2)
b2 = pinns.bspline.BSplineBasis(np.linspace(0,1,2),1)

knots = np.concatenate( ( np.array([[0,1],[0,1],[0,1],[2,2],[3,3]])[...,None] , np.array([[0,0],[1,1],[3,2],[3,2],[3,2]])[...,None] ),-1)
weights = np.array([[1,1],[1,1],[1/np.sqrt(2),1/np.sqrt(2)],[1,1],[1,1]])
knots = np.transpose(knots,[1,0,2])
weights = weights.T
# knots = np.array([[[0, 0],[1, 0]],[[0, 1],[1, 1]],[[0, 2],[1, 2]],[[0, 3],[1, 3]],[[0, 5],[1, 5]]], dtype = np.float64)
# knots = np.transpose(knots,[1,0,2])
# weights = np.ones(knots.shape[:2])

geom = pinns.geometry.PatchNURBS([b2, b1], knots, weights, key)

R = 1
r = 0.2
 
knots = np.array([ [[R,0],[0.5*(R+r),0],[r,0]] , [[R,R],[0.5*(R+r),0.5*(R+r)],[r,r]] , [[0,R],[0,0.5*(R+r)],[0,r]]  ])
weights = np.array([[1,1,1],[1/np.sqrt(2),1/np.sqrt(2),1/np.sqrt(2)],[1,1,1]])

geom = pinns.geometry.PatchNURBS([pinns.bspline.BSplineBasis(np.linspace(0,1,2),2), pinns.bspline.BSplineBasis(np.linspace(0,1,2),2)],knots,weights,key)



class Model(pinns.PINN):
    def __init__(self, rand_key):
        super().__init__()
        self.key = rand_key

        nl = 32
        
        block = stax.serial(stax.FanOut(2),stax.parallel(stax.serial(stax.Dense(nl), stax.Tanh, stax.Dense(nl), stax.Tanh),stax.Dense(nl)),stax.FanInSum)
    
        self.add_neural_network('u',stax.serial(block,block,block,block,block,stax.Dense(1)),(-1,2))
        # self.add_neural_network('u',stax.serial(stax.Dense(nl), stax.Tanh, stax.Dense(nl), stax.Tanh,stax.Dense(nl), stax.Tanh, stax.Dense(nl), stax.Tanh,stax.Dense(nl), stax.Tanh, stax.Dense(nl), stax.Tanh,stax.Dense(1)),(-1,2))
        self.init_points()
        
        
    def init_points(self):
        N = [64,128]
        Knots = np.meshgrid(np.polynomial.legendre.leggauss(N[0])[0]*0.5+0.5, np.polynomial.legendre.leggauss(N[1])[0]*0.5+0.5)
        ys = np.concatenate(tuple([k.flatten()[:,None] for k in Knots]),-1)
        Weights = np.kron(np.polynomial.legendre.leggauss(N[0])[1]*0.5, np.polynomial.legendre.leggauss(N[1])[1]*0.5)
        # ys = np.random.rand(5000,2)*2-1
        # Weights = np.ones((5000))/5000*4
        
        self.points = { 'ys' : ys , 'ws' : Weights}
        DGys = geom._eval_omega(ys)
        Inv = np.linalg.inv(DGys)
        det = np.abs(np.linalg.det(DGys))
        self.points['K'] = np.einsum('mij,mjk,m->mik',Inv,np.transpose(Inv,[0,2,1]),det)
        self.points['omega'] = det
        n = 500
        # p1 = np.concatenate((np.zeros((n,1)),np.random.rand(n,1)),1)
        # p2 = np.concatenate((np.ones((n,1)),np.random.rand(n,1)),1)
        p3 = np.concatenate((np.random.rand(n,1),np.zeros((n,1))),1)
        p4 = np.concatenate((np.random.rand(n,1),np.ones((n,1))),1)
       # self.points['bd'] = np.concatenate((p1,p2,p3,p4),0)
        self.points['bd1'] = p3
        self.points['bd2'] = p4
    def solution(self, ws, x):
        
        u = self.neural_networks['u'](ws['u'],x)
        # v = (jnp.cos(np.pi/2*x[...,0])**2 * jnp.cos(np.pi/2*x[...,1])**2)[...,None]
        v = ((x[...,0] - 1)*(x[...,0] + 0)*(x[...,1] - 1)*(x[...,1] + 0))[...,None]
        # v = ((x[...,1] - 1)*(x[...,1] + 0))[...,None]
        w = 0#(x[...,1][...,None])
        return u*v+w
    
    def loss_pde(self, ws):
        grad = pinns.operators.gradient(lambda x : self.solution(ws,x))(self.points['ys'])
        nu = 1/3
        fval = (lambda x : self.solution(ws,x))(self.points['ys']) 
        
        lpde = 0.5*nu*jnp.dot(jnp.einsum('mi,mij,mj->m',grad,self.points['K'],grad), self.points['ws'])  - jnp.dot(1.0*fval.flatten()*self.points['omega'].flatten(), self.points['ws'])

        return lpde

    def loss(self, ws):
        #lbd = jnp.mean((self.solution(ws,self.points['bd1'])-0)**2)+jnp.mean((self.solution(ws,self.points['bd2'])-1.0)**2)
        lpde = self.loss_pde(ws)

        return lpde#+0*lbd



rnd_key = jax.random.PRNGKey(123)


model = Model(rnd_key)

w0 = model.init_unravel()
weights = w0


loss_compiled = jax.jit(model.loss_handle)
lossgrad_compiled = jax.jit(model.lossgrad_handle)

print('Starting optimization')


def loss_grad(w):
    l, gr = lossgrad_compiled(jnp.array(w))
    return np.array( l.to_py() ), np.array( gr.to_py() ) 

tme = datetime.datetime.now()
#results = jax.scipy.optimize.minimize(loss_grad, x0 = weights_vector, method = 'bfgs', options = {'maxiter': 10})
# result = scipy.optimize.minimize(loss_grad, x0 = w0.to_py(), method = 'BFGS', jac = True, tol = 1e-8, options = {'disp' : True, 'maxiter' : 400}, callback = lambda x: print(loss_compiled(x)))
result = scipy.optimize.minimize(loss_grad, x0 = weights.to_py(), method = 'L-BFGS-B', jac = True, tol = 1e-9, options = {'disp' : True, 'maxiter' : 10000, 'iprint': 1})
tme = datetime.datetime.now() - tme

weights = model.weights_unravel(jnp.array(result.x))

print()
print('Elapsed time', tme)


# opt_init, opt_update, get_params = optimizers.sgd(0.00025)
# opt_state = opt_init(model.weights)
# 
# loss = jax.jit(model.loss)
# for i in range(1000):
#     value, grads = jax.value_and_grad(loss)(get_params(opt_state))
#     opt_state = opt_update(i, grads, opt_state)
#     print(i,value)
# 
# weights = get_params(opt_state)

x,y = np.meshgrid(np.linspace(0,1,100),np.linspace(0,1,100))
ys = np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),1)
xy = geom(ys)

Az = model.solution(weights, ys).reshape(x.shape)

plt.figure()
plt.contourf(xy[:,0].reshape(x.shape), xy[:,1].reshape(x.shape), Az, levels = 64)
# plt.scatter(geom(model.points['ys'])[:,0],geom(model.points['ys'])[:,1],s=1,c='r')
plt.colorbar()

x,y = np.meshgrid(np.linspace(0,1,100),np.linspace(0,1,100))
ys = np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),1)
xy = ys

Az = model.solution(weights, ys).reshape(x.shape)

plt.figure()
plt.contourf(xy[:,0].reshape(x.shape), xy[:,1].reshape(x.shape), Az, levels = 64)
plt.scatter(model.points['ys'][:,0], model.points['ys'][:,1], s=2, c='r')
plt.colorbar()

x,y = np.meshgrid(np.linspace(0,1,32),np.linspace(0,1,32))
ys = np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),1)
xy = (ys)

DGys = np.linalg.det(geom._eval_omega(ys)).reshape(x.shape)

plt.figure()
plt.contourf(xy[:,0].reshape(x.shape), xy[:,1].reshape(x.shape), DGys, levels = 12)
plt.scatter(xy[:,0],xy[:,1],s=1 , c='r')
plt.colorbar()