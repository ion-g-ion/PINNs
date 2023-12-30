import jax
import jax.numpy as jnp
import numpy as np
from jax.example_libraries import stax, optimizers
import matplotlib.pyplot as plt
import datetime
import jax.scipy.optimize
import jax.flatten_util
from typing import Tuple, Callable, Union, Dict, List, Set, TypedDict, Sequence, Any
import functools
from .geometry import PatchNURBS, PatchNURBSParam, PatchConnectivity 
from .functions import jacobian
        

class FunctionSpaceNN():
    __bounds: Tuple[Tuple[float, float]]
    __nn: callable 
    __init: callable
    __d: int 
    __dparam: int
    
    def __init__(self, nn_pair: Tuple[callable], bounds: Tuple[Tuple[float]], nparams: int=0, mask: Tuple[Callable]|None=None):
        
        self.__nn = nn_pair[1]
        self.__init = nn_pair[0]
        self.__bounds = bounds 
        self.__d = len(bounds)
        self.__dparam = nparams
        
    @property 
    def neural_network(self) -> callable:
        return self.__nn
    
    @property
    def bounds(self) -> Tuple[Tuple[float, float]]:
        """
        Return the bounds/definition domain of the NN functions.

        Returns:
            Tuple[Tuple[float, float]]: the intervals.
        """
        return self.__bounds
    
    def init_weights(self, rnd_key: jax.random.KeyArray):
        if self.__dparam == 0:
            return self.__init(rnd_key, (self.__d,))[1]
        else:
            return self.__init(rnd_key, ((-1, self.__d), (-1, self.__dparam)))[1]
    


    def interface_function(self, axis_self: Tuple[int], end_self: Tuple[int], axis_other: Tuple[int], end_other: Tuple[int], bounds_other: Tuple[Tuple[float]], axis_correspondence: Tuple[int, int], decay_fun: callable=lambda x: x) -> callable:
        """
        Return an interface function for the given NN function.

        Args:
            axis_self (Tuple[int]): the axes of the neural network that are fixed.
            end_self (Tuple[int]): the ends where the neural networks are cut. Must take the values 0 or -1.
            axis_other (Tuple[int]): _description_
            end_other (Tuple[int]): _description_
            bounds_other (Tuple[Tuple[float]]): the bounds of the other domain. Tuple of intervals ((a1, b1), (a2,b2), ...).
            decay_fun (callable, optional): the decay function. Suggestions are polynomials of form `x**alpha`. Defaults to lambdax:x.

        Returns:
            callable: the function. Takes as input the weights of the NN, the input of the NN and eventually batch parameters.
        """
        assert isinstance(axis_self, tuple) and all([isinstance(i, int) and i>=0 and i<self.__d for i in axis_self]), "Check the axis_self argument."
        assert isinstance(end_self, tuple) and all([isinstance(i, int) and (i==0 or i==-1) for i in end_self]), "Check the end_self argument."
        assert isinstance(axis_other, tuple) and all([isinstance(i, int) and i>=0 and i<self.__d for i in axis_other]), "Check the axis_other argument."
        assert isinstance(end_other, tuple) and all([isinstance(i, int) and (i==0 or i==-1) for i in end_other]), "Check the end_other argument."
         
        endpositive = np.array([bounds_other[a][e] for a,e in zip(axis_other, end_other)])
        endzero = np.array([bounds_other[a][-1 if e == 0 else 0] for a, e in zip(axis_other, end_other)])
        faux = lambda x: decay_fun((x-endzero)/(endpositive-endzero))
        
        #mask = np.ones((self.__d,))
        #mask[list(axis_self)] = 0
        perm = [a[0] for a in axis_correspondence]
        mask = np.array([(0.0 if k in axis_self else 1.0) for k in range(self.__d)])
        offset = np.zeros((self.__d,))
        offset[list(axis_self)] = np.array([self.__bounds[a][0 if e==0 else 1] for a, e in zip(axis_self, end_self)])
        
        scale_other = np.zeros((self.__d,))
        offset_other = np.zeros((self.__d,))
        scale_self = np.zeros((self.__d,))
        offset_self = np.zeros((self.__d,))
        for k in range(self.__d):
            if axis_correspondence[k][1] == 1: 
                scale_other[k] = 1/(bounds_other[k][1]-bounds_other[k][0])
                offset_other[k] = -bounds_other[k][0]/(bounds_other[k][1]-bounds_other[k][0])
            else:
                scale_other[k] = -1/(bounds_other[k][1]-bounds_other[k][0])
                offset_other[k] = bounds_other[k][1]/(bounds_other[k][1]-bounds_other[k][0])
            
            scale_self[k] = (self.__bounds[k][1]-self.__bounds[k][0])
            offset_self[k] = self.__bounds[k][0]
            
        alpha = scale_other[perm]*scale_self*mask
        beta = offset_other[perm]*scale_self*mask+ offset_self*mask+offset

        fret = lambda ws, x, *args: self.__nn(ws, x[..., perm]*alpha+beta, *args)*jnp.prod(faux(x[...,list(axis_other)]), axis=-1)[...,None]
        
        return fret

def assemble_function(space_this: FunctionSpaceNN, name_this: str, interfaces: dict) -> Callable:

    def f(ws, x, *args):
        acc = space_this.neural_network(ws[name_this], x, *args)
        for c in interfaces:
            acc += interfaces[c](ws[c], x, *args)
        return acc
    return f
        
def connectivity_to_interfaces(spaces: Dict[str, FunctionSpaceNN], connectivity: Sequence[PatchConnectivity], decay_fun: Callable = lambda x: x) -> Dict[str, Callable]:
    ret = dict()
    for name in spaces:
        interfaces = dict()
        for c in connectivity:
            if c['first'] == name:
                axes = c['axis_permutation']
                # permute in this case
                # axes = tuple((a[0], axes[a[0]][1]) for a in axes)
                interfaces[c['second']] = spaces[c['second']].interface_function(c['axis_second'], c['end_second'], c['axis_first'], c['end_first'], spaces[name].bounds, axes, decay_fun )
            elif c['second'] == name:
                axes = c['axis_permutation']
                 # permute in this case
                axes = tuple((a[0], axes[a[0]][1]) for a in axes)
                interfaces[c['first']] = spaces[c['first']].interface_function(c['axis_first'], c['end_first'], c['axis_second'], c['end_second'], spaces[name].bounds, axes, decay_fun )
        ret[name] = assemble_function(spaces[name], name, interfaces)
    return ret
           
class IsogeometricForm():
    __d: int
    __de: int
    __ys: jax.Array 
    __ws: jax.Array
    __xs: jax.Array
    __jac: jax.Array
    __invjac: jax.Array | None
    __ws_param: jax.Array | None 
    __param: jax.Array | None
    __metric: jax.Array | None
    __omega: jax.Array | None
    __invert: float
    
    def __init__(self, d: int, de: int, points_reference: jax.Array, points: jax.Array, jacobian: jax.Array, weights: jax.Array, parameters: jax.Array | None, weights_parameters: jax.Array | None, inverted: float=1.0):
        self.__ys = points_reference
        self.__xs = points
        self.__ws = weights 
        self.__d = d
        self.__de = de
        self.__jac = jacobian
        self.__invjac = None
        self.__metric = None 
        self.__invert = inverted

    @property 
    def points_reference(self) -> jax.Array:
        return self.__ys
    
    @property 
    def points_physical(self) -> jax.Array:
        return self.__xs

    @property 
    def weights(self) -> jax.Array:
        return self.__ws
    
    @property
    def geometry_jacobian(self) -> jax.Array:
        return self.__jac 
    
    @property 
    def geometry_jacobian_inverse(self) -> jax.Array:
        if self.__invjac is None:
            self.__invjac = jnp.linalg.inv(self.__jac)
        return self.__invjac
    
    @property
    def metric_coefficient(self) -> jax.Array:
        
        if self.__metric is None:
            self.__compute_metric()
            
        return self.__metric
    
    def  jacobian_transformation(self, jac: jax.Array) -> jax.Array:
        """
        Transform the jacobian in the reference domain to the physical.
        
        Args:
            jac (jax.Array): the jacobian in the reference domain evluated on the integration points.

        Returns:
            jax.Array: the result.
        """
        
        return jnp.einsum('...ij,...jk->...ik', jac, self.geometry_jacobian_inverse)
    
    def jacbian_physical(self, f: Callable) -> jax.Array:
        """
        Compute the jacobian in the physical domain and evaluate it along the quadrature points.

        Args:
            f (Callable): the function.

        Returns:
            jax.Array: the reusult.
        """
            
        pass 
    
    def jacobian_reference(self, f: Callable) -> jax.Array:
        """
        Compute the jacobian of a function in the reference domain.

        Args:
            f (Callable): _description_

        Returns:
            jax.Array: _description_
        """
        return jacobian(f)(self.__ys)
    
    def __compute_metric(self):
        if self.__d == self.__de:
            # volume integral
            if self.__metric is None:
                self.__metric = self.__invert * (jnp.linalg.det(self.__jac))[...,None]
        elif self.__d == 2:
            if self.__metric is None:
                self.__metric = self.__invert * jnp.cross(self.__jac[...,0], self.__jac[...,1])
        elif self.__d == 1 and self.__de == 3:
            if self.__metric is None:
                self.__metric = self.__invert * self.__jac
        elif self.__d == 1 and self.__de == 2:
            if self.__metric is None:
                self.__metric = self.__invert * jnp.einsum('mn,kn->km', np.array([[0.0,1.0],[-1.0,0.0]]), self.__jac)
        else:
            raise Exception("Metric not really defined for the manifold dimension %d and embedding dimension %d"%(self.__d, self.__de))    
        
    
    def dx(self, vector_field: bool=True) -> jax.Array: 
        if self.__metric is None:
            self.__compute_metric()
        if vector_field == False or self.__d == self.__de:
            return (jnp.linalg.norm(self.__metric, axis=-1)*self.__ws)
        else:
            return self.__metric * self.__ws[...,None]

                
        
    
    # @property
    # def volume_metric(self):
    #     if self.__omega is None:
    #         self.__omega = jnp.abs(jnp.linalg.det(self.__jac))
    #     return self.__omega
    
    # @property 
    # def dv(self) -> jax.Array:
    #     if self.__omega is None:
    #         self.__omega = jnp.abs(jnp.linalg.det(self.__jac))
    #     return self.__omega*self.__ws
    
    # @property 
    # def ds(self) -> jax.Array:
    #     if self.__orientation is None:
    #         if self.__d == 2 and self.__de==3:
    #             self.__orientation = jnp.cross(self.__jac[...,0], self.__jac[...,1])*self.__invert

    #     
    
    # @property
    # def ds_vec(self) -> jax.Array:
    #     pass
    
    # @property
    # def dl(self) -> jax.Array:
    #     pass 
    
    # @property 
    # def dl_vec(self) -> jax.Array:
    #     pass
            
class PINN():
    __patches: Dict[str, PatchNURBS | PatchNURBSParam] | None
    weights: Dict 
     
    def __init__(self, patches: Dict[str, PatchNURBS | PatchNURBSParam] | None): 
        self.neural_networks = {}
        self.neural_networks_initializers = {}
        self.__patches = patches
    
    @property 
    def patches(self) -> Dict[str, PatchNURBS | PatchNURBSParam] | None:
        return self.__patches
    
    def add_neural_network(self, name, ann, input_shape):
        
        self.neural_networks[name] = ann[1]
        self.neural_networks_initializers[name] = ann[0]
        self.weights[name] = ann[0](self.key, input_shape)[1]

    def add_neural_network_param(self, name, ann, input_shape):
        
        self.neural_networks[name] = lambda w,x,p : ann[1](w,(x,p))
        self.neural_networks_initializers[name] = ann[0]
        self.weights[name] = ann[0](self.key, input_shape)[1]

    def add_trainable_parameter(self, name, shape):

        self.weights[name] = jax.random.normal(self.key, shape)

    def init_unravel(self):
        
        weights_vector, weights_unravel = jax.flatten_util.ravel_pytree(self.weights)
        self.weights_unravel = weights_unravel
        return weights_vector
         
    def points_MonteCarlo(self, N, key, facets: List[Dict] = [], parameter_sampler: Callable | None=None):        

        points = {}

        for name in self.__patches:
            
            bounds = np.array(self.__patches[name].domain)
            ys = jax.random.uniform(key ,(N,self.__patches[name].d))*(bounds[:,1]-bounds[:,0]) + bounds[:,0]
            Weights = jnp.ones((N,))*np.prod(bounds[:,1]-bounds[:,0])/ys.shape[0]
             
            DG = self.__patches[name].GetJacobian(ys)
            xs = self.__patches[name](ys)
            
            points[name] = IsogeometricForm(self.__patches[name].d, self.__patches[name].dembedding, ys, xs, DG, Weights, None, None, 1)
            #omega, DGys, Gr, K = self.__patches[name].GetMetricTensors(ys)
            #points[name] = {'pts': ys, 'ws': Weights, 'dV': omega, 'Jac': DGys, 'JacInv': Gr, 'InnerGradProd': K}

        for facet in facets:
            patch_name = facet['patch']
            label = facet['label']
            axis = facet['axis']
            end = facet['end']
            if 'n' in facet:
                n = facet['n']
            else:
                n = N         
                
            selector = [i for i in range(self.__patches[patch_name].d) if i != axis]
            bounds = np.array(self.__patches[patch_name].domain)
            bounds[axis,:] = bounds[axis,end]
            ys = jax.random.uniform(key ,(n,self.__patches[patch_name].d))*(bounds[:, 1]-bounds[:, 0]) + bounds[:, 0]
            Weights = jnp.ones((n,))*np.prod(bounds[selector,1]-bounds[selector,0])/ys.shape[0]
            
            DG = self.__patches[patch_name].GetJacobian(ys)[...,:,selector]
            xs = self.__patches[patch_name](ys)
            
            points[label] = IsogeometricForm(self.__patches[patch_name].d-1, self.__patches[patch_name].dembedding, ys, xs, DG, Weights, None, None, 1)
        
            
        return points
    
    def loss(self, training_parameters: Any):
        pass
    
    def train(self, method = 'ADAM'):
        pass

  
    def loss_handle(self, w):
        ws = self.weights_unravel(w)
        l = self.loss(ws)
        return l


    def lossgrad_handle(self, w, *args):
        ws = self.weights_unravel(w)
        
        l = self.loss(ws, *args)
        gr = jax.grad(self.loss)(ws, *args)
        
        gr,_ = jax.flatten_util.ravel_pytree(gr)
        return l, gr


def DirichletMask(nn_tuple: Tuple[Callable], out_dims: int | Tuple[int], domain: list[tuple[float, float]], conditions: list[dict], alpha = 1.0) -> Tuple[Callable, Callable]:
    """
    Factory method for a Dirichlet adapted NN.
    Right now only works with 0 Dirichlet.
    
    Args:
        nn_tuple (Tuple[Callable]): the init and call tuple. Follows `jax.stax` convention.
        out_dims (int | Tuple[int]): number of output dimensions.
        domain (list[tuple[float, float]]): the domain where the NN is defined.
        conditions (list[dict]): the boundary conditions. A dictionary of tyoe `{'dim': int, 'end': +-1}`. Deciding across which axis and what end the boundary condition is applied.
        alpha (float, optional): decay. Defaults to 1.0.

    Returns:
        Tuple[Callable, Callable]: resulting initializer and call functions.
    """
    def init_fun(rng, input_shape):
        return nn_tuple[0](rng, input_shape)

    def apply_fun(params, inputs, **kwargs):
        res = 1
        for c in conditions:
            res = res * ((inputs[..., c['dim']]-(domain[c['dim']][0] if c['end'] == 0 else domain[c['dim']][1]))**alpha)[...,None]
        return jnp.tile(res, out_dims)*nn_tuple[1](params, inputs, **kwargs)
    return init_fun, apply_fun
