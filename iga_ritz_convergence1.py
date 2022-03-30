
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
    def __init__(self, rand_key, nl = 16, N = [64,64], num_blocks = 4):
        super().__init__()
        self.key = rand_key


        
        block = stax.serial(stax.FanOut(2),stax.parallel(stax.serial(stax.Dense(nl), stax.Tanh, stax.Dense(nl), stax.Tanh),stax.Dense(nl)),stax.FanInSum)
    
        self.add_neural_network('u1',stax.serial(*([block]*num_blocks),stax.Dense(1)),(-1,2))
        self.add_neural_network('u2',stax.serial(*([block]*num_blocks),stax.Dense(1)),(-1,2))
        self.add_neural_network('u12',stax.serial(stax.Dense(nl),stax.Tanh,stax.Dense(nl),stax.Tanh,stax.Dense(nl),stax.Tanh,stax.Dense(1)),(-1,1))
        # self.add_neural_network('u',stax.serial(stax.Dense(nl), stax.Tanh, stax.Dense(nl), stax.Tanh,stax.Dense(nl), stax.Tanh, stax.Dense(nl), stax.Tanh,stax.Dense(nl), stax.Tanh, stax.Dense(nl), stax.Tanh,stax.Dense(1)),(-1,2))
        self.init_points(N)
        
        self.interface12 = interface_function2d(1,0.0,1.0,self.neural_networks['u12'])
        self.interface21 = interface_function2d(1,1.0,0.0,self.neural_networks['u12'])
        
        self.eps1 = 1/100
        self.eps2 = 1/2
        self.U = 10
        
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
        w = self.interface21(ws['u12'],x) + (1-x[...,1][...,None])*self.U
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


def simmulation( nl = 16, N = [64,64], num_blocks = 2):
    print('Width ',nl,' blocks ',num_blocks)
    results = {'width' : nl, 'blocks' : num_blocks, 'integration_grid_size': N}
    model = Model(rnd_key, nl = nl, N = N, num_blocks = num_blocks)

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
    result = scipy.optimize.minimize(loss_grad, x0 = weights.to_py(), method = 'L-BFGS-B', jac = True, tol = 1e-11, options = {'disp' : False, 'maxiter' : 4000, 'iprint': 0})
    tme = datetime.datetime.now() - tme

    weights = model.weights_unravel(jnp.array(result.x))

    
    results['parameters'] = result.x.size
    results['time'] = tme.total_seconds()
    results['fun_calls'] = result.nfev
    results['fval'] = result.fun
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

    a2 = model.U/(np.log(Ro)-np.log(R)+model.eps2/model.eps1*np.log(R/r))
    uref1 = lambda x: a2*model.eps2/model.eps1*np.log(np.sqrt(x[:,0]**2+x[:,1]**2)/r)
    uref2 = lambda x: a2*(np.log(np.sqrt(x[:,0]**2+x[:,1]**2)/R)+model.eps2/model.eps1*np.log(R/r))

    integral1 = np.sum((model.solution1(weights,model.points['ys']).flatten()-uref1(geom1(model.points['ys'])))**2*model.points['omega1']*model.points['ws']) 
    integral2 = np.sum((model.solution2(weights,model.points['ys']).flatten()-uref2(geom2(model.points['ys'])))**2*model.points['omega2']*model.points['ws']) 
    inttotal = np.sum((model.solution1(weights,model.points['ys']).flatten())**2*model.points['omega1']*model.points['ws']) + np.sum((model.solution2(weights,model.points['ys']).flatten())**2*model.points['omega2']*model.points['ws'])

    error = np.sqrt(integral1+integral2)/np.sqrt(inttotal)
    results['error_L2'] = error
    
    print('Relative error is %e' % error)
    
    return results

results = []
for nl in [3,4,5,6,7,8,9]:
    res = simmulation(nl = nl)
    results.append(res)
    
import pandas as pd
df = pd.DataFrame.from_dict(results)
print(df)

results = []
for N in [30,40,50,60,70,80,90,100,120]:
    res = simmulation(N = [N,N])
    results.append(res)
    
import pandas as pd
df = pd.DataFrame.from_dict(results)
print(df)