
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


R = 1
Ro = 1.5
r = 0.2
 
knots = np.array([ [[R,0],[0.5*(R+r),0],[r,0]] , [[R,R],[0.5*(R+r),0.5*(R+r)],[r,r]] , [[0,R],[0,0.5*(R+r)],[0,r]]  ])
weights = np.array([[1,1,1],[1/np.sqrt(2),1/np.sqrt(2),1/np.sqrt(2)],[1,1,1]])

geom1 = pinns.geometry.PatchNURBS([pinns.bspline.BSplineBasis(np.linspace(0,1,2),2), pinns.bspline.BSplineBasis(np.linspace(0,1,2),2)],knots,weights,key)

knots = np.array([ [[Ro,0],[0.5*(Ro+R),0],[R,0]] , [[Ro,Ro],[0.5*(Ro+R),0.5*(Ro+R)],[R,R]] , [[0,Ro],[0,0.5*(Ro+R)],[0,R]]  ])
weights = np.array([[1,1,1],[1/np.sqrt(2),1/np.sqrt(2),1/np.sqrt(2)],[1,1,1]])
geom2 = pinns.geometry.PatchNURBS([pinns.bspline.BSplineBasis(np.linspace(0,1,2),2), pinns.bspline.BSplineBasis(np.linspace(0,1,2),2)],knots,weights,key)


def interface_function2d(nd, endpositive, endzero, nn):

    faux = lambda x: ((x-endzero)**2/(endpositive-endzero)**2)
    if nd == 0:
        fret = lambda ws, x: (nn(ws, x[...,1][...,None]).flatten()*faux(x[...,0]))[...,None]
    else:
        fret = lambda ws, x: (nn(ws, x[...,0][...,None]).flatten()*faux(x[...,1]))[...,None]
    return fret

class Model(pinns.PINN):
    def __init__(self, rand_key):
        super().__init__()
        self.key = rand_key

        N = [64,128]
        nl = 4
        
        block = stax.serial(stax.FanOut(2),stax.parallel(stax.serial(stax.Dense(nl), stax.Tanh, stax.Dense(nl), stax.Tanh),stax.Dense(nl)),stax.FanInSum)
    
        self.add_neural_network('u1',stax.serial(block,block,block,block,block,stax.Dense(1)),(-1,2))
        self.add_neural_network('u2',stax.serial(block,block,block,block,block,stax.Dense(1)),(-1,2))
        self.add_neural_network('u12',stax.serial(stax.Dense(nl),stax.Tanh,stax.Dense(nl),stax.Tanh,stax.Dense(1)),(-1,1))
        # self.add_neural_network('u',stax.serial(stax.Dense(nl), stax.Tanh, stax.Dense(nl), stax.Tanh,stax.Dense(nl), stax.Tanh, stax.Dense(nl), stax.Tanh,stax.Dense(nl), stax.Tanh, stax.Dense(nl), stax.Tanh,stax.Dense(1)),(-1,2))
        self.init_points(N)
        
        self.interface12 = interface_function2d(1,0.0,1.0,self.neural_networks['u12'])
        self.interface21 = interface_function2d(1,1.0,0.0,self.neural_networks['u12'])
        
        self.eps1 = 1/6
        self.eps2 = 1/2
        
    def init_points(self, N):
        
        Knots = np.meshgrid(np.polynomial.legendre.leggauss(N[0])[0]*0.5+0.5, np.polynomial.legendre.leggauss(N[1])[0]*0.5+0.5)
        ys = np.concatenate(tuple([k.flatten()[:,None] for k in Knots]),-1)
        Weights = np.kron(np.polynomial.legendre.leggauss(N[0])[1]*0.5, np.polynomial.legendre.leggauss(N[1])[1]*0.5)
        # ys = np.random.rand(5000,2)*2-1
        # Weights = np.ones((5000))/5000*4
        
        self.points = { 'ys' : ys , 'ws' : Weights}
        DGys = geom1._eval_omega(ys)
        Inv = np.linalg.inv(DGys)
        det = np.abs(np.linalg.det(DGys))
        self.points['K1'] = np.einsum('mij,mjk,m->mik',Inv,np.transpose(Inv,[0,2,1]),det)
        self.points['omega1'] = det
       
        DGys = geom2._eval_omega(ys)
        Inv = np.linalg.inv(DGys)
        det = np.abs(np.linalg.det(DGys))
        self.points['K2'] = np.einsum('mij,mjk,m->mik',Inv,np.transpose(Inv,[0,2,1]),det)
        self.points['omega2'] = det
        
        
    def solution1(self, ws, x):
        
        u = self.neural_networks['u1'](ws['u1'],x)
        # v = (jnp.cos(np.pi/2*x[...,0])**2 * jnp.cos(np.pi/2*x[...,1])**2)[...,None]
        #v = ((x[...,0] - 1)*(x[...,0] + 0)*(x[...,1] - 1)*(x[...,1] + 0))[...,None]
        v = ((x[...,1] - 1)*(x[...,1] + 0))[...,None]
        
        w =  self.interface12(ws['u12'],x)
        
        return u*v+w
    
    def solution2(self, ws, x):
        
        u = self.neural_networks['u2'](ws['u2'],x)
        # v = (jnp.cos(np.pi/2*x[...,0])**2 * jnp.cos(np.pi/2*x[...,1])**2)[...,None]
        #v = ((x[...,0] - 1)*(x[...,0] + 0)*(x[...,1] - 1)*(x[...,1] + 0))[...,None]
        v = ((x[...,1] - 1)*(x[...,1] + 0))[...,None]
        w = self.interface21(ws['u12'],x) + (1-x[...,1][...,None])
        return u*v+w
    
    
    def loss_pde(self, ws):
        grad1 = pinns.operators.gradient(lambda x : self.solution1(ws,x))(self.points['ys'])[...,0,:]
        grad2 = pinns.operators.gradient(lambda x : self.solution2(ws,x))(self.points['ys'])[...,0,:]
        
        
        
        lpde1 = 0.5*self.eps1*jnp.dot(jnp.einsum('mi,mij,mj->m',grad1,self.points['K1'],grad1), self.points['ws'])  
        lpde2 = 0.5*self.eps2*jnp.dot(jnp.einsum('mi,mij,mj->m',grad2,self.points['K2'],grad2), self.points['ws'])  
        return lpde1+lpde2

    def loss(self, ws):
        lpde = self.loss_pde(ws)
        return lpde



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
result = scipy.optimize.minimize(loss_grad, x0 = weights.to_py(), method = 'L-BFGS-B', jac = True, tol = 1e-9, options = {'disp' : True, 'maxiter' : 1000, 'iprint': 1})
tme = datetime.datetime.now() - tme

weights = model.weights_unravel(jnp.array(result.x))

print()
print('Elapsed time', tme)


# opt_init, opt_update, get_params = optimizers.sgd(0.0025)
# opt_state = opt_init(model.weights)
# 
# loss = jax.jit(model.loss)
# for i in range(1000):
#     value, grads = jax.value_and_grad(loss)(get_params(opt_state))
#     opt_state = opt_update(i, grads, opt_state)
#     print(i,value)
# 
# weights = get_params(opt_state)
# weights = model.weights

x,y = np.meshgrid(np.linspace(0,1,100),np.linspace(0,1,100))
ys = np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),1)
xy1 = geom1(ys)
xy2 = geom2(ys)

u1 = model.solution1(weights, ys).reshape(x.shape)
u2 = model.solution2(weights, ys).reshape(x.shape)

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(xy1[:,0].reshape(x.shape), xy1[:,1].reshape(x.shape), u1, cmap ='viridis', edgecolor =None)
ax.plot_surface(xy2[:,0].reshape(x.shape), xy2[:,1].reshape(x.shape), u2, cmap ='viridis', edgecolor =None)
plt.show()


t = np.linspace(0,1,1000)
xy1 = geom1(np.concatenate((t[:,None]*0,t[:,None]),1))
xy2 = geom2(np.concatenate((t[:,None]*0,t[:,None]),1))

uline1 = model.solution1(weights,np.concatenate((t[:,None]*0+0.5,t[:,None]),1)).flatten()
uline2 = model.solution2(weights,np.concatenate((t[:,None]*0+0.5,t[:,None]),1)).flatten()

a2 = 1/(np.log(Ro)-np.log(R)+model.eps2/model.eps1*np.log(R/r))
uref1 = lambda x: a2*model.eps2/model.eps1*np.log(np.sqrt(x[:,0]**2+x[:,1]**2)/r)
uref2 = lambda x: a2*(np.log(np.sqrt(x[:,0]**2+x[:,1]**2)/R)+model.eps2/model.eps1*np.log(R/r))
    
plt.figure()
plt.plot(xy1[:,0],uline1)
plt.plot(xy2[:,0],uline2)
plt.plot(xy1[:,0],uref1(xy1))
plt.plot(xy2[:,0],uref2(xy2))
# plt.plot(xy[:,0],np.log(xy[:,0]/R)/np.log(r/R))

plt.figure()
plt.plot(xy1[:,0],np.log10(np.abs(uline1-uref1(xy1))))
plt.plot(xy2[:,0],np.log10(np.abs(uline2-uref2(xy2))))