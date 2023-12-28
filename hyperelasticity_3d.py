#%% Imports
import os 
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries import stax, optimizers
import matplotlib.pyplot as plt
import pyvista as pv
import pinns 
import datetime
import jax.scipy.optimize
import jax.flatten_util
import scipy
import scipy.optimize

from jax.config import config
config.update("jax_enable_x64", False)
rnd_key = jax.random.PRNGKey(1234)
np.random.seed(14124)

def create_geometry(key, scale = 1):
    R = 2
    r = 1
    Rp = 2
    d = 0.5
    
    knots = np.array( [ [[[r,0,0], [r,0,r], [0,0,r]], [[r,d,0], [r,d,r], [0,d,r]]], [[[R,0,0], [R,0,R], [0,0,R]], [[R,d,0], [R,d,R], [0,d,R]]] ])
    weights = np.ones(knots.shape[:3])
    weights[:,:,1] = 1/np.sqrt(2)
    
    basis1 = pinns.functions.BSplineBasisJAX(np.array([-1,1]),1)
    basis2 = pinns.functions.BSplineBasisJAX(np.array([-1,1]),1)
    basis3 = pinns.functions.BSplineBasisJAX(np.array([-1,1]),2)

    geom1 = pinns.geometry.PatchNURBSParam([basis1, basis2, basis3], knots, weights, 0, 3, key)
    
    knots = np.array( [[[[0,0,r],[-(R-r),0,r]], [[0,d,r],[-(R-r),d,r]]], [[[0,0,R],[-(R-r),0,R]], [[0,d,R],[-(R-r),d,R]]]] )
    weights = np.ones(knots.shape[:3])
    
    basis1 = pinns.functions.BSplineBasisJAX(np.array([-1,1]),1)
    basis2 = pinns.functions.BSplineBasisJAX(np.array([-1,1]),1)
    basis3 = pinns.functions.BSplineBasisJAX(np.array([-1,1]),1)

    geom2 = pinns.geometry.PatchNURBSParam([basis1, basis2, basis3], knots, weights, 0, 3, key)
    
    knots = np.array( [[[[r+Rp,-R,r], [-(R-r),-R,r], [-(R-r),0,r]], [[r+Rp,-r,r], [0,-r,r], [0,0,r]]], [[[r+Rp,-R,R], [-(R-r),-R,R], [-(R-r),0,R]], [[r+Rp,-r,R], [0,-r,R], [0,0,R]]]] )
    knots = np.transpose(knots,[0,2,1,3])
    knots = knots[:,:,::-1,:]
    weights = np.ones(knots.shape[:3])
    weights[:,1,:] = 1/np.sqrt(2)
    
    basis1 = pinns.functions.BSplineBasisJAX(np.array([-1,1]),1)
    basis3 = pinns.functions.BSplineBasisJAX(np.array([-1,1]),1)
    basis2 = pinns.functions.BSplineBasisJAX(np.array([-1,1]),2)

    geom3 = pinns.geometry.PatchNURBSParam([basis1, basis2, basis3], knots, weights, 0, 3, key)
    
    
    knots = np.array( [[[[r+Rp,R+d,r], [-(R-r),R+d,r], [-(R-r),d,r]], [[r+Rp,r+d,r], [0,r+d,r], [0,d,r]]], [[[r+Rp,R+d,R], [-(R-r),R+d,R], [-(R-r),d,R]], [[r+Rp,r+d,R], [0,r+d,R], [0,d,R]]]] )
    knots = np.transpose(knots,[0,2,1,3])
    knots = knots[:,::-1,...]
    knots = knots[:,:,::-1,:]
    weights = np.ones(knots.shape[:3])
    weights[:,1,:] = 1/np.sqrt(2)
    
    
    basis1 = pinns.functions.BSplineBasisJAX(np.array([-1,1]),1)
    basis2 = pinns.functions.BSplineBasisJAX(np.array([-1,1]),2)
    basis3 = pinns.functions.BSplineBasisJAX(np.array([-1,1]),1)

    geom4 = pinns.geometry.PatchNURBSParam([basis1, basis2, basis3], knots, weights, 0, 3, key)
    
    return  geom1, geom2, geom3, geom4

    
geom1, geom2, geom3, geom4 = create_geometry(rnd_key)
names = ['obj1', 'obj2', 'obj3', 'obj4']
geoms = {names[0]: geom1, names[1]: geom2, names[2]: geom3, names[3]: geom4}

connectivity = [
    pinns.geometry.PatchConnectivity(first='obj1', second='obj2', axis_first=(2,), axis_second=(2,), end_first=(-1,), end_second=(0,), axis_permutation=((0,1),(1,1),(2,1))),
    pinns.geometry.PatchConnectivity(first='obj2', second='obj3', axis_first=(1,), axis_second=(1,), end_first=(0,), end_second=(-1,), axis_permutation=((0,1),(1,1),(2,1))),
    pinns.geometry.PatchConnectivity(first='obj2', second='obj4', axis_first=(1,), axis_second=(1,), end_first=(-1,), end_second=(0,), axis_permutation=((0,1),(1,1),(2,1))),
    pinns.geometry.PatchConnectivity(first='obj1', second='obj3', axis_first=(1,2), axis_second=(1,2), end_first=(0,-1), end_second=(-1,0), axis_permutation=((0,1),(1,1),(2,1))),
    pinns.geometry.PatchConnectivity(first='obj1', second='obj4', axis_first=(1,2), axis_second=(1,2), end_first=(-1,-1), end_second=(0,0), axis_permutation=((0,1),(1,1),(2,1))),
]

#connectivity = pinns.geometry.match_patches(geoms)
#assert len(connectivity) == 5

pv_objects = [pinns.extras.plot(g, {'y0': lambda y: y[...,0], 'y1': lambda y: y[...,1], 'y2': lambda y: y[...,2]}, N= 16) for g in geoms.values()]

obj = pv_objects[0].merge(pv_objects[1])
obj = obj.merge(pv_objects[2])
obj = obj.merge(pv_objects[3])

pv_objects[0].save('obj1.vtk')
pv_objects[1].save('obj2.vtk')
pv_objects[2].save('obj3.vtk')
pv_objects[3].save('obj4.vtk')

try:
    plotter = pv.Plotter(window_size=(600, 400))
    plotter.background_color = 'w'
    plotter.enable_anti_aliasing()
    plotter.add_mesh(obj, show_edges=True)
    #plotter.show()
except:
    print('Cannot plot')

nl = 32
acti =  stax.elementwise(lambda x: jax.nn.leaky_relu(x)**2)
w_init = jax.nn.initializers.normal()

block_first = stax.serial(stax.FanOut(2),stax.parallel(stax.serial(stax.Dense(nl,W_init = w_init), acti, stax.Dense(nl,W_init = w_init), acti),stax.Dense(nl,W_init = w_init)),stax.FanInSum)
block = stax.serial(stax.FanOut(2),stax.parallel(stax.serial(stax.Dense(nl,W_init = w_init), acti, stax.Dense(nl,W_init = w_init), acti),stax.Dense(nl,W_init = w_init)),stax.FanInSum)
nn = stax.serial(block_first,block, stax.Dense(3))

space_bc = pinns.FunctionSpaceNN(pinns.DirichletMask(nn, 3, geom1.domain, [{'dim': 2, 'end': 0}]), ((-1,1), (-1,1), (-1,1))) 
space = pinns.FunctionSpaceNN(nn,((-1,1), (-1,1), (-1,1)))


class Pinn(pinns.PINN):
    
    def __init__(self):
          
        self.weights = {names[0]: space_bc.init_weights(rnd_key), names[1]: space.init_weights(rnd_key), names[2]: space.init_weights(rnd_key), names[3]: space.init_weights(rnd_key)}
        self.solutions = pinns.connectivity_to_interfaces({names[0]: space_bc, names[1]: space, names[2]: space, names[3]: space}, connectivity)
        
        E = 0.02e5
        nu = 0.1
        self.E = E
        self.nu = nu
        
        self.lamda = E*nu/(1+nu)/(1-2*nu)
        self.mu = E/2/(1+nu)

        rho = 0.2
        g = 9.81
        self.rho = rho
        
        self.f = np.array([0,0,-g*rho]) 
        self.energy = lambda F,C,J,params: params[0]*jnp.sum(F**2, axis=(-2,-1)) + params[1]*jnp.abs(J)**2*jnp.sum(jnp.linalg.inv(F)**2, axis=(-1,-2)) + params[2]*J**2 - params[3]*jnp.log(jnp.abs(J))+params[4]
        self.energy = lambda F,C,J,params: 0.5*self.mu*(C[...,0,0]+C[...,1,1]+C[...,2,2]-3)-self.mu*jnp.log(jnp.abs(J))+0.5*self.lamda*jnp.log(jnp.abs(J))**2
        
        self.a = 0.5*self.mu
        self.b = 0.0
        self.c = 0.0
        self.d = self.mu
        self.e = -1.5*self.mu

        super(Pinn, self).__init__({names[0]: geom1, names[1]: geom2, names[2]: geom3, names[3]: geom4})
   
 
    def loss(self, training_parameters, points):
        names = ['obj1', 'obj2', 'obj3', 'obj4']

        jacs = [pinns.functions.jacobian(lambda x : self.solutions[n](training_parameters, x))(points[n].points_reference) for n in names]
        jacs_x = [points[names[i]].jacobian_transformation(jacs[i]) for i in range(4)]
        Fs = [jnp.eye(3)+jacs_x[i] for i in range(4)]
        Cs = [jnp.einsum('mij,mik->mjk', Fs[i], Fs[i]) for i in range(4)]
        
        dets = [jnp.linalg.det(Fs[i]) for i in range(4)]
         
        Es = [jnp.dot(self.energy(Fs[i], Cs[i], dets[i], [self.a, self.b,self.c,self.d,self.e]), points[names[i]].dx()) for i in range(4)]
        rhss = [jnp.dot(dets[i] * jnp.einsum('k,mk->m', self.f, self.solutions[names[i]](training_parameters, points[names[i]].points_reference)), points[names[i]].dx()) for i in range(4)] 

        return sum(Es) - sum(rhss)
    
        
model = Pinn()  
dev = jax.devices('gpu')[0] if len(jax.devices('gpu'))>0 else jax.devices('cpu')[0]

opt_type = 'ADAM'

if opt_type == 'ADAM':

    batch_size = 1000

    # get_compiled = jax.jit(lambda key: model.get_points_MC(batch_size, key), device = dev)
    # %time pts = get_compiled(jax.random.PRNGKey(1235))
    # %time pts = get_compiled(jax.random.PRNGKey(1111))

    lr_opti = optimizers.piecewise_constant([2000,3000,4000,5000,7000], [0.005, 0.005/2, 0.005/4, 0.005/8,0.005/16,0.005/32])
    #lr_opti = optimizers.piecewise_constant([2000,3000,4000,5000], [0.005/2, 0.005/4, 0.005/8,0.005/16,0.005/32])
    # lr_opti = optimizers.piecewise_constant([7000], [0.01/2, 0.001])
    opt_init, opt_update, get_params = optimizers.adam(lr_opti)

    opt_state = opt_init(model.weights)
    weights_init = model.weights
    
    # get initial parameters
    params = get_params(opt_state)

    loss_grad = jax.jit(lambda ws, pts: (model.loss(ws, pts), jax.grad(model.loss)(ws, pts)), device = dev)

    def step(params, opt_state, key):
        # points = model.get_points_MC(5000)
        points = model.points_MonteCarlo(batch_size, key)
        loss = model.loss(params, points)
        grads = jax.grad(model.loss)(params, points)
        #loss, grads = loss_grad(params, points)
        opt_state = opt_update(0, grads, opt_state)

        params = get_params(opt_state)
        
        return params, opt_state, loss

    step_compiled = jax.jit(step, device = dev)
    step_compiled(params, opt_state, rnd_key)

    n_epochs = 10000

    hist = []
    hist_weights = []
    
    # min_loss = 10000
    tme = datetime.datetime.now()
    for k in range(n_epochs):    
        params, opt_state, loss = step_compiled(params, opt_state, jax.random.PRNGKey(k%1000+0*np.random.randint(1000)))
        
        hist.append(loss)
        
        if k%50 == 0:
            hist_weights.append(params.copy())
        print('Epoch %d/%d - loss value %e'%(k+1, n_epochs, loss))
        
    # update params
    model.weights = params
    weights = params
    tme = datetime.datetime.now() - tme
    print('Elapsed time ', tme)

    
pv_objects = [pinns.extras.plot(geoms[n], {'displacement': lambda y: model.solutions[n](weights, y)}, N= 32) for n in geoms]

obj = pv_objects[0].merge(pv_objects[1])
obj = obj.merge(pv_objects[2])
obj = obj.merge(pv_objects[3])
obj.save('solution.vtk')

try:
    plotter = pv.Plotter(window_size=(600, 400))
    plotter.background_color = 'w'
    plotter.enable_anti_aliasing()
    plotter.add_mesh(obj, show_edges=True)
except:
    print('Cannot plot')