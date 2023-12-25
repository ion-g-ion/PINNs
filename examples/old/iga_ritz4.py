

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


R = 2.5
r = 0.5
 
knots = np.array([ [[-r,-r],[-r,0],[-r,r]] , [[0,-r],[0,0],[0,r]] , [[r,-r],[r,0],[r,r]] ])
weights = np.ones((3,3))
geom1 = pinns.geometry.PatchNURBS([pinns.bspline.BSplineBasis(np.linspace(-1,1,2),2), pinns.bspline.BSplineBasis(np.linspace(-1,1,2),2)],knots,weights,key)

knots = np.array([ [[r,-r],[r,0],[r,r]], [[R,-r],[R,(R+r)*np.tan(np.pi/8)-r],[R/np.sqrt(2),R/np.sqrt(2)]]  ])
weights = np.ones((2,3))
weights[1,1] = np.sin(np.pi/8)
geom2 = pinns.geometry.PatchNURBS([pinns.bspline.BSplineBasis(np.linspace(-1,1,2),1), pinns.bspline.BSplineBasis(np.linspace(-1,1,2),2)],knots,weights,key)

knots = np.array([ [[-r,r],[-r,R]], [[0,r],[(R+r)*np.tan(np.pi/8)-r,R]] ,[[r,r],[R/np.sqrt(2),R/np.sqrt(2)]]  ])
weights = np.ones((3,2))
weights[1,1] = np.sin(np.pi/8)
geom3 = pinns.geometry.PatchNURBS([pinns.bspline.BSplineBasis(np.linspace(-1,1,2),2), pinns.bspline.BSplineBasis(np.linspace(-1,1,2),1)],knots,weights,key)


plt.figure()
xy,_ = geom1.importance_sampling(10000)
plt.scatter(xy[:,0],xy[:,1],s=1)
xy,_ = geom2.importance_sampling(10000)
plt.scatter(xy[:,0],xy[:,1],s=1)
xy,_ = geom3.importance_sampling(10000)
plt.scatter(xy[:,0],xy[:,1],s=1)

x1,_ = geom2[-1,:].importance_sampling(1000)
x2,_ = geom2[:,-1].importance_sampling(1000)
x3,_ = geom2[1,:].importance_sampling(1000)
x4,_ = geom2[:,1].importance_sampling(1000)


plt.figure()
plt.scatter(x1[:,0],x1[:,1],s=1,c='k')
plt.scatter(x2[:,0],x2[:,1],s=1,c='k')
plt.scatter(x3[:,0],x3[:,1],s=1,c='k')
plt.scatter(x4[:,0],x4[:,1],s=1,c='k')
plt.text(geom2(np.array([[-1,-1]]))[0,0],geom2(np.array([[-1,-1]]))[0,1],"(-1,-1)")
plt.text(geom2(np.array([[-1,1]]))[0,0],geom2(np.array([[-1,1]]))[0,1],"(-1,1)")
plt.text(geom2(np.array([[1,-1]]))[0,0],geom2(np.array([[1,-1]]))[0,1],"(1,-1)")
plt.text(geom2(np.array([[1,1]]))[0,0],geom2(np.array([[1,1]]))[0,1],"(1,1)")

x1,_ = geom1[-1,:].importance_sampling(1000)
x2,_ = geom1[:,-1].importance_sampling(1000)
x3,_ = geom1[1,:].importance_sampling(1000)
x4,_ = geom1[:,1].importance_sampling(1000)

plt.figure()
plt.scatter(x1[:,0],x1[:,1],s=1,c='k')
plt.scatter(x2[:,0],x2[:,1],s=1,c='k')
plt.scatter(x3[:,0],x3[:,1],s=1,c='k')
plt.scatter(x4[:,0],x4[:,1],s=1,c='k')
plt.text(geom1(np.array([[-1,-1]]))[0,0],geom1(np.array([[-1,-1]]))[0,1],"(-1,-1)")
plt.text(geom1(np.array([[-1,1]]))[0,0],geom1(np.array([[-1,1]]))[0,1],"(-1,1)")
plt.text(geom1(np.array([[1,-1]]))[0,0],geom1(np.array([[1,-1]]))[0,1],"(1,-1)")
plt.text(geom1(np.array([[1,1]]))[0,0],geom1(np.array([[1,1]]))[0,1],"(1,1)")
plt.show()

x1,_ = geom3[-1,:].importance_sampling(1000)
x2,_ = geom3[:,-1].importance_sampling(1000)
x3,_ = geom3[1,:].importance_sampling(1000)
x4,_ = geom3[:,1].importance_sampling(1000)

plt.figure()
plt.scatter(x1[:,0],x1[:,1],s=1,c='k')
plt.scatter(x2[:,0],x2[:,1],s=1,c='k')
plt.scatter(x3[:,0],x3[:,1],s=1,c='k')
plt.scatter(x4[:,0],x4[:,1],s=1,c='k')
plt.text(geom3(np.array([[-1,-1]]))[0,0],geom3(np.array([[-1,-1]]))[0,1],"(-1,-1)")
plt.text(geom3(np.array([[-1,1]]))[0,0],geom3(np.array([[-1,1]]))[0,1],"(-1,1)")
plt.text(geom3(np.array([[1,-1]]))[0,0],geom3(np.array([[1,-1]]))[0,1],"(1,-1)")
plt.text(geom3(np.array([[1,1]]))[0,0],geom3(np.array([[1,1]]))[0,1],"(1,1)")
plt.show()


def interface_function2d(nd, endpositive, endzero, nn):

    faux = lambda x: ((x-endzero)**1/(endpositive-endzero)**1)
    if nd == 0:
        fret = lambda ws, x: (nn(ws, x[...,1][...,None]).flatten()*faux(x[...,0]))[...,None]
    else:
        fret = lambda ws, x: (nn(ws, x[...,0][...,None]).flatten()*faux(x[...,1]))[...,None]
    return fret

class Model(pinns.PINN):
    def __init__(self, rand_key):
        super().__init__()
        self.key = rand_key

        N = [64,64]
        nl = 8
        acti = stax.Tanh
        block = stax.serial(stax.FanOut(2),stax.parallel(stax.serial(stax.Dense(nl), acti, stax.Dense(nl), acti),stax.Dense(nl)),stax.FanInSum)
    
        self.add_neural_network('u1',stax.serial(block,block,block,stax.Dense(1)),(-1,2)) # iron
        self.add_neural_network('u2',stax.serial(block,block,block,stax.Dense(1)),(-1,2)) # air 
        self.add_neural_network('u3',stax.serial(block,block,block,stax.Dense(1)),(-1,2)) # copper
        self.add_neural_network('u12',stax.serial(stax.Dense(nl), acti, stax.Dense(nl), acti, stax.Dense(1)),(-1,1))
        self.add_neural_network('u13',stax.serial(stax.Dense(nl), acti, stax.Dense(nl), acti, stax.Dense(1)),(-1,1))
        self.add_neural_network('u23',stax.serial(stax.Dense(nl), acti, stax.Dense(nl), acti, stax.Dense(1)),(-1,1))
        self.add_trainable_parameter('u123',(1,))
        self.init_points(N)
        
        #self.interface12 = interface_function2d(1,1.0,0.0,self.neural_networks['u12'])
        #self.interface21 = interface_function2d(0,1.0,0.0,self.neural_networks['u12'])
        #self.interface23 = interface_function2d(1,1.0,0.0,self.neural_networks['u23'])
        #self.interface32 = interface_function2d(1,1.0,0.0,self.neural_networks['u23'])
        #self.interface13 = interface_function2d(0,1.0,0.0,self.neural_networks['u13'])
        #self.interface31 = interface_function2d(0,1.0,0.0,self.neural_networks['u13'])
        #self.interface14 = interface_function2d(1,0.0,1.0,self.neural_networks['u14'])
        #self.interface41 = interface_function2d(1,1.0,0.0,self.neural_networks['u14'])
        #self.interface34 = interface_function2d(1,0.0,1.0,self.neural_networks['u34'])
        #self.interface43 = interface_function2d(0,1.0,0.0,self.neural_networks['u34'])
        self.mu0 = 1/1.0
        self.mur = 1/1
        self.J0 = 100

    def init_points(self, N):
        
        Knots = np.meshgrid(np.polynomial.legendre.leggauss(N[0])[0], np.polynomial.legendre.leggauss(N[1])[0])
        ys = np.concatenate(tuple([k.flatten()[:,None] for k in Knots]),-1)
        Weights = np.kron(np.polynomial.legendre.leggauss(N[0])[1], np.polynomial.legendre.leggauss(N[1])[1])
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
        
        DGys = geom3._eval_omega(ys)
        Inv = np.linalg.inv(DGys)
        det = np.abs(np.linalg.det(DGys))
        self.points['K3'] = np.einsum('mij,mjk,m->mik',Inv,np.transpose(Inv,[0,2,1]),det)
        self.points['omega3'] = det
       


    def solution1(self, ws, x):
        # iron
        u = self.neural_networks['u1'](ws['u1'],x)
        v = ((1-x[...,0])*(1-x[...,1])*(1+x[...,0])*(1+x[...,1]))[...,None]
        w = self.neural_networks['u12'](ws['u12'],x[...,1][...,None])*((1+x[...,0] )*(1 - x[...,1])*(1 + x[...,1]))[...,None] + self.neural_networks['u13'](ws['u13'],x[...,0][...,None])*((1-x[...,0])*(1+x[...,1] )*(1+x[...,0] ))[...,None]
        w = w + ws['u123']*( (1+x[...,0])*(1+x[...,1]) )[...,None]
        return u*v+w

    def solution2(self, ws, x):
        
        u = self.neural_networks['u2'](ws['u2'],x)
        v = ((1-x[...,0])*(1-x[...,1])*(1+x[...,0])*(1+x[...,1]))[...,None]
        w = self.neural_networks['u12'](ws['u12'],x[...,1][...,None])*((1- x[...,0])*(1 - x[...,1])*(1 + x[...,1]))[...,None] + self.neural_networks['u23'](ws['u23'],x[...,0][...,None])*((1-x[...,0] )*(1+x[...,0] )*(1+x[...,1]))[...,None]
        w = w + ws['u123']*( (1+x[...,1])*(1-x[...,0]) )[...,None]
        return u*v+w
    
    def solution3(self, ws, x):
        
        u = self.neural_networks['u3'](ws['u3'],x)
        v = ((1-x[...,0])*(1-x[...,1])*(1+x[...,0])*(1+x[...,1]))[...,None]
        w =  self.neural_networks['u13'](ws['u13'],x[...,0][...,None])*((1-x[...,0])*(1-x[...,1])*(1+x[...,0] ))[...,None] + self.neural_networks['u23'](ws['u23'],x[...,1][...,None])*((1-x[...,1])*(1+x[...,1])*(1+x[...,0]))[...,None]
        w = w + ws['u123']*( (x[...,0]+1)*(1-x[...,1]) )[...,None]
        return u*v + w
        


    def loss_pde(self, ws):
        grad1 = pinns.operators.gradient(lambda x : self.solution1(ws,x))(self.points['ys'])
        grad2 = pinns.operators.gradient(lambda x : self.solution2(ws,x))(self.points['ys'])
        grad3 = pinns.operators.gradient(lambda x : self.solution3(ws,x))(self.points['ys'])
        
        lpde1 = 0.5*self.mu0*jnp.dot(jnp.einsum('mi,mij,mj->m',grad1,self.points['K1'],grad1), self.points['ws']) + jnp.dot(self.J0*self.solution1(ws,self.points['ys']).flatten()*self.points['omega1']  ,self.points['ws'])
        lpde2 = 0.5*(self.mu0*self.mur)*jnp.dot(jnp.einsum('mi,mij,mj->m',grad2,self.points['K2'],grad2), self.points['ws'])   + jnp.dot(self.J0*self.solution2(ws,self.points['ys']).flatten()*self.points['omega2']  ,self.points['ws'])
        lpde3 = 0.5*self.mu0*jnp.dot(jnp.einsum('mi,mij,mj->m',grad3,self.points['K3'],grad3), self.points['ws'])  + jnp.dot(self.J0*self.solution3(ws,self.points['ys']).flatten()*self.points['omega3']  ,self.points['ws'])
        return lpde1+lpde2+lpde3

    def loss(self, ws):
        lpde = self.loss_pde(ws)
        return lpde

rnd_key = jax.random.PRNGKey(111)
model = Model(rnd_key)
w0 = model.init_unravel()
weights = model.weights 


loss_compiled = jax.jit(model.loss_handle)
lossgrad_compiled = jax.jit(model.lossgrad_handle)

print('Starting optimization')

def loss_grad(w):
    l, gr = lossgrad_compiled(jnp.array(w))
    return np.array( l.to_py() ), np.array( gr.to_py() ) 

tme = datetime.datetime.now()
#results = jax.scipy.optimize.minimize(loss_grad, x0 = weights_vector, method = 'bfgs', options = {'maxiter': 10})
# result = scipy.optimize.minimize(loss_grad, x0 = w0.to_py(), method = 'BFGS', jac = True, tol = 1e-8, options = {'disp' : True, 'maxiter' : 2000}, callback = None)
result = scipy.optimize.minimize(loss_grad, x0 = w0.to_py(), method = 'L-BFGS-B', jac = True, tol = 1e-9, options = {'disp' : True, 'maxiter' : 2000, 'iprint': 1})
tme = datetime.datetime.now() - tme

weights = model.weights_unravel(jnp.array(result.x))

x,y = np.meshgrid(np.linspace(-1,1,200),np.linspace(-1,1,200))
ys = np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),1)
xy1 = geom1(ys)
xy2 = geom2(ys)
xy3 = geom3(ys)


u1 = model.solution1(weights, ys).reshape(x.shape)
u2 = model.solution2(weights, ys).reshape(x.shape)
u3 = model.solution3(weights, ys).reshape(x.shape)


fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(xy1[:,0].reshape(x.shape), xy1[:,1].reshape(x.shape), u1, cmap ='viridis', vmin = min([u1.min(),u2.min(),u3.min()]), vmax = max([u1.max(),u2.max(),u3.max()]), edgecolor =None)
ax.plot_surface(xy2[:,0].reshape(x.shape), xy2[:,1].reshape(x.shape), u2, cmap ='viridis', vmin = min([u1.min(),u2.min(),u3.min()]), vmax = max([u1.max(),u2.max(),u3.max()]), edgecolor =None)
ax.plot_surface(xy3[:,0].reshape(x.shape), xy3[:,1].reshape(x.shape), u3, cmap ='viridis', vmin = min([u1.min(),u2.min(),u3.min()]), vmax = max([u1.max(),u2.max(),u3.max()]), edgecolor =None)
ax.view_init(90,0)
plt.show()

