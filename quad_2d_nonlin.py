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
rnd_key = jax.random.PRNGKey(1234)

#%% Geometry parametrizations

def create_geometry(key, scale = 1):
    scale = scale
    Nt = 24                                                                
    lz = 40e-3                                                             
    Do = 72e-3                                                            
    Di = 51e-3                                                            
    hi = 13e-3                                                             
    bli = 3e-3                                                             
    Dc = 3.27640e-2                                                           
    hc = 7.55176e-3                                                           
    ri = 20e-3                                                           
    ra = 18e-3                                                           
    blc = hi-hc                                                           
    rm = (Dc*Dc+hc*hc-ri*ri)/(Dc*np.sqrt(2)+hc*np.sqrt(2)-2*ri)                 
    R = rm-ri
    O = np.array([rm/np.sqrt(2),rm/np.sqrt(2)])
    alpha1 = -np.pi*3/4       
    alpha2 = np.math.asin((hc-rm/np.sqrt(2))/R)
    alpha = np.abs(alpha2-alpha1)
    
    A = np.array([[O[0] - ri/np.sqrt(2), O[1] - ri/np.sqrt(2)], [O[0] - Dc, O[1] - hc]])
    b = np.array([[A[0,0]*ri/np.sqrt(2)+A[0,1]*ri/np.sqrt(2)],[A[1,0]*Dc+A[1,1]*hc]])
    C = np.linalg.solve(A,b)
    
    knots1 = np.array([[Do,Do * np.tan(np.pi/8)],[Do/np.sqrt(2),Do/np.sqrt(2)],[rm/np.sqrt(2),rm/np.sqrt(2)],[ri/np.sqrt(2),ri/np.sqrt(2)]])
    #knots2 = np.array([[Dc,hc],[Dc+blc,hi],[Di-bli,hi],[Di,hi-bli],[Di,0]])
    knots2 = np.array([[Di,hi-bli],[Di-bli,hi],[Dc+blc,hi],[Dc,hc]])
    knots3 = (knots1+knots2)/2
    knots3[-1,:] = C.flatten()
    knots = np.concatenate((knots1[None,...],knots3[None,...],knots2[None,...]),0)
    weights = np.ones(knots.shape[:2])
    weights[1,-1] = np.sin((np.pi-alpha)/2)
    basis2 = pinns.bspline.BSplineBasisJAX(np.array([-1,-0.33,0.33,1]),1)
    basis1 = pinns.bspline.BSplineBasisJAX(np.array([-1,1]),2)

    geom1 = pinns.geometry.PatchNURBSParam([basis1, basis2], knots, weights, 0, 2, key)
   
    knots2 = np.array([ [ [Dc,0],[Dc+blc,0],[Di-bli,0],[Di,0] ] , [[Dc,hc],[Dc+blc,hi],[Di-bli,hi],[Di,hi-bli]] ]) 
    knots2 = knots2[:,::-1,:]
    weights = np.ones(knots2.shape[:2])
    
    basis1 = pinns.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
    basis2 = pinns.bspline.BSplineBasisJAX(np.array([-1,-0.33,0.33,1]),1)

    geom2 = pinns.geometry.PatchNURBSParam([basis1, basis2], knots2, weights, 0, 2, key)
   
    knots = np.array([ [ [0,0] , [Dc/2,0] , [Dc,0] ] , [ [ri/np.sqrt(2),ri/np.sqrt(2)] , [C[0,0],C[1,0]] , [Dc,hc] ]])
    
    basis1 = pinns.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
    basis2 = pinns.bspline.BSplineBasisJAX(np.array([-1,1]),2)
    
    weights = np.ones(knots.shape[:2])
    weights[1,1] = np.sin((np.pi-alpha)/2)
    geom3 = pinns.geometry.PatchNURBSParam([basis1, basis2], knots, weights, 0, 2, key)

    knots1 = np.array([[Do,0],[Do,Do * np.tan(np.pi/8)]])
    knots2 = np.array([[Di,0],[Di,hi-bli]])
    knots3 = (knots1+knots2)/2
    knots = np.concatenate((knots1[None,...],knots3[None,...],knots2[None,...]),0)
    weights = np.ones(knots.shape[:2])

    basis2 = pinns.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
    basis1 = pinns.bspline.BSplineBasisJAX(np.array([-1,1]),2)

    geom4 = pinns.geometry.PatchNURBSParam([basis1, basis2], knots, weights, 0, 2, key)
    return  geom1, geom3, geom2, geom4

#%% Instantiate geometry parametrizations

geom1, geom2, geom3, geom4 = create_geometry(rnd_key)


pts,_ = geom1.importance_sampling(10000)

plt.figure()
plt.scatter(pts[:,0], pts[:,1], s = 1)

pts,_ = geom2.importance_sampling(10000)
plt.scatter(pts[:,0],pts[:,1], s = 1)

pts,_ = geom3.importance_sampling(10000)
plt.scatter(pts[:,0],pts[:,1], s = 1)

pts,_ = geom4.importance_sampling(10000)
plt.scatter(pts[:,0],pts[:,1], s = 1)
plt.show()

#%% Define the model

def interface_function2d(nd, endpositive, endzero, nn):

    faux = lambda x: ((x-endzero)**1/(endpositive-endzero)**1)
    if nd == 0:
        fret = lambda ws, x: (nn(ws, x[...,1][...,None]).flatten()*faux(x[...,0]))[...,None]
    else:
        fret = lambda ws, x: (nn(ws, x[...,0][...,None]).flatten()*faux(x[...,1]))[...,None]
    return fret

def jump_function2d(nd, pos_y, nn):

    faux = lambda x: jnp.exp(-4.0*jnp.abs(x-pos_y))
    if nd == 1:
        fret = lambda ws, x: (nn(ws, x[...,1][...,None]).flatten()*faux(x[...,0]))[...,None]
    else:
        fret = lambda ws, x: (nn(ws, x[...,0][...,None]).flatten()*faux(x[...,1]))[...,None]
    return fret

def ExpHat(x, scale = 0.1):
    return jnp.exp(-jnp.abs(x)/scale)

class Model(pinns.PINN):
    def __init__(self, rand_key):
        super().__init__()
        self.key = rand_key

        N = [32,32]
        nl = 16
        acti = stax.Tanh
        acti =  stax.elementwise(lambda x: jax.nn.leaky_relu(x)**2)
        acti1 = stax.elementwise(lambda x: jax.nn.leaky_relu(x+1)**2)
        acti2 = stax.elementwise(lambda x: jax.nn.leaky_relu(x+0.33)**2)
        acti3 = stax.elementwise(lambda x: jax.nn.leaky_relu(x-0.33)**2)
        acti4 = stax.elementwise(lambda x: jnp.exp(-1.0*jnp.abs(x)))
        
        block_first = stax.serial(stax.FanOut(2),stax.parallel(stax.serial(stax.Dense(nl), acti, stax.Dense(nl), acti),stax.Dense(nl)),stax.FanInSum)
        block = stax.serial(stax.FanOut(2),stax.parallel(stax.serial(stax.Dense(nl), acti, stax.Dense(nl), acti),stax.Dense(nl)),stax.FanInSum)
        block2 = lambda n: stax.serial(stax.FanOut(2),stax.parallel(stax.serial(stax.Dense(n), acti, stax.Dense(n), acti),stax.Dense(n)),stax.FanInSum)
        block3 = stax.serial(stax.FanOut(2),stax.parallel(stax.serial(stax.Dense(nl), acti3, stax.Dense(nl), acti2),stax.Dense(nl)),stax.FanInSum)
        
        self.add_neural_network('u1',stax.serial(block_first,block,block, block, stax.Dense(1)),(-1,2)) # iron
        self.add_neural_network('u4',stax.serial(block_first,block,block, block, stax.Dense(1)),(-1,2)) # iron 2
        self.add_neural_network('u2',stax.serial(block_first,block,block, block, stax.Dense(1)),(-1,2)) # air 
        self.add_neural_network('u3',stax.serial(block_first,block,block, block, stax.Dense(1)),(-1,2)) # copper
        self.add_neural_network('u12',stax.serial(block_first, block, block, stax.Dense(1)),(-1,1))
        self.add_neural_network('u13',stax.serial(block_first, block, block, stax.Dense(1)),(-1,1))
        # self.add_neural_network('u13',stax.serial(stax.Dense(1000), acti4, stax.Dense(1)),(-1,1))
        self.add_neural_network('u23',stax.serial(block_first, block, block, stax.Dense(1)),(-1,1))
        self.add_neural_network('u14',stax.serial(block_first, block, block, stax.Dense(1)),(-1,1))
        self.add_neural_network('u34',stax.serial(block_first, block, block, stax.Dense(1)),(-1,1))
        self.add_neural_network('u1_0.3',stax.serial(block_first, block, block, stax.Dense(1)),(-1,1))
        self.add_neural_network('u1_0.7',stax.serial(block_first, block, block, stax.Dense(1)),(-1,1))
        self.add_trainable_parameter('u123',(1,))
        self.add_trainable_parameter('u134',(1,))
        self.add_trainable_parameter('u13_p0.33',(1,))
        self.add_trainable_parameter('u13_n0.33',(1,))
        
        
        self.interface12 = interface_function2d(1,1.0,-1.0,self.neural_networks['u12'])
        self.interface21 = interface_function2d(0,1.0,-1.0,self.neural_networks['u12'])
        self.interface23 = interface_function2d(1,1.0,-1.0,self.neural_networks['u23'])
        self.interface32 = interface_function2d(1,1.0,-1.0,self.neural_networks['u23'])
        self.interface13 = interface_function2d(0,1.0,-1.0,self.neural_networks['u13'])
        self.interface31 = interface_function2d(0,1.0,-1.0,self.neural_networks['u13'])
        self.interface14 = interface_function2d(1,-1.0,1.0,self.neural_networks['u14'])
        self.interface41 = interface_function2d(1,1.0,-1.0,self.neural_networks['u14'])
        self.interface34 = interface_function2d(1,-1.0,1.0,self.neural_networks['u34'])
        self.interface43 = interface_function2d(0,1.0,-1.0,self.neural_networks['u34'])

        self.jump1 = jump_function2d(0, -0.33, self.neural_networks['u1_0.3'])
        self.jump2 = jump_function2d(0,  0.33, self.neural_networks['u1_0.7'])

        self.mu0 = 0.001
        self.mur = 2000
        self.J0 =  1000000

        self.k1 = 0.001
        self.k2 = 1.65/5000
        self.k3 = 0.5

        self.points = self.get_points_MC(100000, self.key)
        
    # def get_points(self, N, quad = 'MC'):        

    #     points = {}

    #     if quad == 'MC':
    #         ys = np.random.rand(N,2)*2-1
    #         Weights = np.ones((N,))*4/ys.shape[0]

    #     # ys = np.meshgrid(np.polynomial.legendre.leggauss(N[0])[0], np.polynomial.legendre.leggauss(N[1])[0])
    #     # ys = np.concatenate((ys[0].flatten()[:,None], ys[1].flatten()[:,None]), -1)
    #     # Weights = np.kron(np.polynomial.legendre.leggauss(N[0])[1], np.polynomial.legendre.leggauss(N[1])[1]).flatten()
    #     if quad=='SG':
    #         ys,Weights = cp.quadrature.sparse_grid(10,cp.J(cp.Uniform(-1,1),cp.Uniform(-1,1)),rule=["fejer_2", "fejer_2"])
    #         Weights = Weights*4
    #         ys = np.transpose(ys)

    #     if quad == 'TP': ys, Weights = pinns.geometry.tensor_product_integration(geom1.basis, N)
    #     points['ys1'] = ys
    #     points['ws1'] = Weights
    #     DGys = geom1._eval_omega(ys)
    #     Inv = np.linalg.inv(DGys)
    #     det = np.abs(np.linalg.det(DGys))
    #     points['K1'] = np.einsum('mij,mjk,m->mik',Inv,np.transpose(Inv,[0,2,1]),det)
    #     points['omega1'] = det
    #    
    #     if quad == 'TP': ys, Weights = pinns.geometry.tensor_product_integration(geom2.basis, N)
    #     points['ys2'] = ys
    #     points['ws2'] = Weights
    #     DGys = geom2._eval_omega(ys)
    #     Inv = np.linalg.inv(DGys)
    #     det = np.abs(np.linalg.det(DGys))
    #     points['K2'] = np.einsum('mij,mjk,m->mik',Inv,np.transpose(Inv,[0,2,1]),det)
    #     points['omega2'] = det
    #     
    #     if quad == 'TP': ys, Weights = pinns.geometry.tensor_product_integration(geom3.basis, N)
    #     points['ys3'] = ys
    #     points['ws3'] = Weights
    #     DGys = geom3._eval_omega(ys)
    #     Inv = np.linalg.inv(DGys)
    #     det = np.abs(np.linalg.det(DGys))
    #     points['K3'] = np.einsum('mij,mjk,m->mik',Inv,np.transpose(Inv,[0,2,1]),det)
    #     points['omega3'] = det
    #    
    #     if quad == 'TP': ys, Weights = pinns.geometry.tensor_product_integration(geom4.basis, N)
    #     points['ys4'] = ys
    #     points['ws4'] = Weights
    #     DGys = geom4._eval_omega(ys)
    #     Inv = np.linalg.inv(DGys)
    #     det = np.abs(np.linalg.det(DGys))
    #     points['K4'] = np.einsum('mij,mjk,m->mik',Inv,np.transpose(Inv,[0,2,1]),det)
    #     points['omega4'] = det

    #     return points

    def get_points_MC(self, N, key):        

        points = {}


        ys = jax.random.uniform(key ,(N,2))*2-1
        Weights = jnp.ones((N,))*4/ys.shape[0]
        # ys = np.array(jax.random.uniform(self.key, (N,2)))*2-1
        # Weights = jnp.ones((N,))*4/ys.shape[0]


        points['ys1'] = ys
        points['ws1'] = Weights
        points['omega1'], points['G1'], points['K1'] = geom1.GetMetricTensors(ys)
       
        points['ys2'] = ys
        points['ws2'] = Weights
        points['omega2'], points['G2'], points['K2'] = geom2.GetMetricTensors(ys)
        
        points['ys3'] = ys
        points['ws3'] = Weights
        points['omega3'], points['G3'], points['K3'] = geom3.GetMetricTensors(ys)
       
        points['ys4'] = ys
        points['ws4'] = Weights
        points['omega4'], points['G4'], points['K4'] = geom4.GetMetricTensors(ys)

        return points


    def solution1(self, ws, x):
        # iron
        alpha = 2
        u = self.neural_networks['u1'](ws['u1'],x) + self.jump1(ws['u1_0.3'], x) + self.jump2(ws['u1_0.7'], x)
        v = ((1-x[...,0])*(x[...,0] + 1)*(1-x[...,1])*(x[...,1]+1))[...,None]
        w =  self.interface12(ws['u12'],x)*((1-x[...,0])*(x[...,0] + 1))[...,None] + (self.interface13(ws['u13'],x)+ExpHat(x[...,1]+0.33)[...,None]*ws['u13_n0.33']+ExpHat(x[...,1]-0.33)[...,None]*ws['u13_p0.33'])*(1-x[...,1])[...,None]*(x[...,1] + 1)[...,None] +  self.interface14(ws['u14'],x) * ((1-x[...,0])*(x[...,0] + 1))[...,None]
        w = w + ws['u123']*( (x[...,0]+1) * (x[...,1]+1) )[...,None]**alpha + ws['u134'] *  ( (x[...,0] + 1)*(1-x[...,1]) )[...,None]**alpha
        return u*v+w

    def solution2(self, ws, x):
        alpha = 2
        u = self.neural_networks['u2'](ws['u2'],x)
        v = ((1-x[...,1])*(x[...,1] + 1)*(1-x[...,0]))[...,None]
        w = self.interface21(ws['u12'],x)*((1-x[...,1])*(x[...,1] + 1))[...,None] + self.interface23(ws['u23'],x)*(1-x[...,0])[...,None]
        w = w + ws['u123']*( (x[...,0]+1) * (x[...,1]+1) )[...,None]**alpha
        return u*v+w
    
    def solution3(self, ws, x):
        alpha = 2
        u = self.neural_networks['u3'](ws['u3'],x)
        v = ((1-x[...,1])*(x[...,1] + 1)*(1-x[...,0]))[...,None]
        w =  self.interface32(ws['u23'],x)*(1-x[...,0])[...,None]+(self.interface31(ws['u13'],x)+ExpHat(x[...,1]+0.33)[...,None]*ws['u13_n0.33']+ExpHat(x[...,1]-0.33)[...,None]*ws['u13_p0.33'])*((1-x[...,1])*(x[...,1] + 1))[...,None]+self.interface34(ws['u34'],x)*(1-x[...,0])[...,None]
        w = w + ws['u123']*( (x[...,0]+1) * (x[...,1]+1) )[...,None]**alpha + ws['u134'] *  ( (x[...,0] +1)*(1-x[...,1]) )[...,None]**alpha
        return u*v + w
        
    def solution4(self, ws, x):
        alpha = 2
        u = self.neural_networks['u4'](ws['u4'],x)
        v = ((1-x[...,0])*(x[...,0] + 1)*(1-x[...,1]))[...,None]
        w = self.interface41(ws['u14'],x)*((1-x[...,0])*(x[...,0] + 1))[...,None]+self.interface43(ws['u34'],x)*((1-x[...,1]))[...,None]
        w = w + ws['u134'] *  ( (x[...,0]+1) * (x[...,1]+1) )[...,None]**alpha
        return u*v+w

    def nu_model(self, grad_a):
        b2 = grad_a[...,0]**2+grad_a[...,1]**2
        return self.k1*jnp.exp(self.k2*b2)+self.k3
    def nu_model(self, b2):
       
        return self.k1*jnp.exp(self.k2*b2)+self.k3
    
    def loss_pde(self, ws, points):
        grad1 = pinns.operators.gradient(lambda x : self.solution1(ws,x))(points['ys1'])[...,0,:]
        grad2 = pinns.operators.gradient(lambda x : self.solution2(ws,x))(points['ys2'])[...,0,:]
        grad3 = pinns.operators.gradient(lambda x : self.solution3(ws,x))(points['ys3'])[...,0,:]
        grad4 = pinns.operators.gradient(lambda x : self.solution4(ws,x))(points['ys4'])[...,0,:]
        
        grad1x = jnp.einsum('mij,mj->mi',points['G1'],grad1)
        grad4x = jnp.einsum('mij,mj->mi',points['G4'],grad4)
        
        # lpde1 = 0.5*1/(self.mu0*self.mur)*jnp.dot(jnp.einsum('mi,mij,mj->m',grad1,self.points['K1'],grad1), self.points['ws1']) 
        # lpde2 = 0.5*1/self.mu0*jnp.dot(jnp.einsum('mi,mij,mj->m',grad2,self.points['K2'],grad2), self.points['ws2'])  
        # lpde3 = 0.5*1/self.mu0*jnp.dot(jnp.einsum('mi,mij,mj->m',grad3,self.points['K3'],grad3), self.points['ws3'])  - jnp.dot(self.J0*self.solution3(ws,self.points['ys3']).flatten()*self.points['omega3']  ,self.points['ws3'])
        # lpde4 = 0.5*1/(self.mu0*self.mur)*jnp.dot(jnp.einsum('mi,mij,mj->m',grad4,self.points['K4'],grad4), self.points['ws4']) 
        bi1 = jnp.einsum('mi,mij,mj->m',grad1,points['K1'],grad1)
        bi4 = jnp.einsum('mi,mij,mj->m',grad4,points['K4'],grad4)
        lpde1 = 0.5*(self.mu0)*jnp.dot(self.nu_model(bi1)*bi1, points['ws1']) 
        lpde2 = 0.5*jnp.dot(jnp.einsum('mi,mij,mj->m',grad2,points['K2'],grad2), points['ws2'])  
        lpde3 = 0.5*jnp.dot(jnp.einsum('mi,mij,mj->m',grad3,points['K3'],grad3), points['ws3'])  - self.mu0*jnp.dot(self.J0*self.solution3(ws,points['ys3']).flatten()*points['omega3']  ,points['ws3'])
        lpde4 = 0.5*(self.mu0)*jnp.dot(self.nu_model(bi4)*bi4, points['ws4'])
        return lpde1+lpde2+lpde3+lpde4

    def loss(self, ws, pts):
        lpde = self.loss_pde(ws, pts)
        return lpde
    

#%% Setup model and train it

rnd_key = jax.random.PRNGKey(1235)
model = Model(rnd_key)
w0 = model.init_unravel()
weights = model.weights 

dev = jax.devices()[1]

# loss_compiled = jax.jit(model.loss_handle, device = jax.devices()[0])
# lossgrad_compiled = jax.jit(model.lossgrad_handle, device = jax.devices()[0])
# 
# def loss_grad(w):
#     l, gr = lossgrad_compiled(jnp.array(w))
#     return np.array( l.to_py() ), np.array( gr.to_py() )

opt_type = 'ADAM'
batch_size = 2000

get_compiled = jax.jit(lambda key: model.get_points_MC(batch_size, key), device = dev)
%time pts = get_compiled(jax.random.PRNGKey(1235))
%time pts = get_compiled(jax.random.PRNGKey(1111))

opt_init, opt_update, get_params = optimizers.adam(step_size=0.001)

opt_state = opt_init(weights)

# get initial parameters
params = get_params(opt_state)

loss_grad = jax.jit(lambda ws, pts: (model.loss(ws, pts), jax.grad(model.loss)(ws, pts)), device = dev)

def step(params, opt_state, key):
    # points = model.get_points_MC(5000)
    points = model.get_points_MC(batch_size, key)
    loss, grads = loss_grad(params, points)
    opt_state = opt_update(0, grads, opt_state)

    params = get_params(opt_state)
    
    return params, opt_state, loss

step_compiled = jax.jit(step, device = dev)
step_compiled(params, opt_state, rnd_key)

n_epochs = 5000

tme = datetime.datetime.now()
for k in range(n_epochs):    
    params, opt_state, loss = step_compiled(params, opt_state, jax.random.PRNGKey(np.random.randint(32131233123)))
    
    print('Epoch %d/%d - loss value %e'%(k+1, n_epochs, loss))
# update params
model.weights = params
weights = params
tme = datetime.datetime.now() - tme
print('Elapsed time ', tme)

x,y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
ys = np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),1)
xy1 = geom1(ys)
xy2 = geom2(ys)
xy3 = geom3(ys)
xy4 = geom4(ys)

u1 = model.solution1(weights, ys).reshape(x.shape)
u2 = model.solution2(weights, ys).reshape(x.shape)
u3 = model.solution3(weights, ys).reshape(x.shape)
u4 = model.solution4(weights, ys).reshape(x.shape)

plt.figure(figsize = (20,12))
ax = plt.gca()
plt.contourf(xy1[:,0].reshape(x.shape), xy1[:,1].reshape(x.shape), u1, levels = 100, vmin = min([u1.min(),u2.min(),u3.min(),u4.min()]), vmax = max([u1.max(),u2.max(),u3.max(),u4.max()]))
plt.contourf(xy2[:,0].reshape(x.shape), xy2[:,1].reshape(x.shape), u2, levels = 100, vmin = min([u1.min(),u2.min(),u3.min(),u4.min()]), vmax = max([u1.max(),u2.max(),u3.max(),u4.max()]))
plt.contourf(xy3[:,0].reshape(x.shape), xy3[:,1].reshape(x.shape), u3, levels = 100, vmin = min([u1.min(),u2.min(),u3.min(),u4.min()]), vmax = max([u1.max(),u2.max(),u3.max(),u4.max()]))
plt.contourf(xy4[:,0].reshape(x.shape), xy4[:,1].reshape(x.shape), u4, levels = 100, vmin = min([u1.min(),u2.min(),u3.min(),u4.min()]), vmax = max([u1.max(),u2.max(),u3.max(),u4.max()]))
plt.colorbar()
plt.xlabel(r'$x_1$ [m]')
plt.ylabel(r'$x_2$ [m]')