import jax
import jax.numpy as jnp
import numpy as np
from jax.example_libraries import stax, optimizers
import matplotlib.pyplot as plt
import datetime
import jax.scipy.optimize
import jax.flatten_util
from typing import Tuple, Callable, Union
import functools

        
class FunctionNN():
    _bounds: Tuple[Tuple[int]]
    _nn: callable
    _init: callable
    _d: int # input dimension
    _dparam:int
    
    def __init__(self, nn, nn_init, bounds: Tuple[Tuple[int]], n_params: int=0):
        self._bounds = bounds
        self._nn = nn 
        self._init = nn_init 
        self._d = len(bounds)
        self._dparam = n_params
        
    def __call__(self, ws, x, *params) -> jax.Array:
        return self._nn(ws, x, *params)
    
    def face_function(self, axis_self, end_self, axis_other, end_other, bounds_other: Tuple[Tuple[int]], decay_fun: callable=lambda x: x) -> callable:
                
        endpositive = bounds_other[axis_other][end_other]
        endzero = bounds_other[axis_other][-1 if end_other == 0 else 0]
        faux = lambda x: decay_fun((x-endzero)/(endpositive-endzero))
        
        mask = np.ones((self._d,))
        mask[axis_self] = 0
        offset = np.zeros((self._d,))
        offset[axis_self] = self._bounds[axis_self][0 if end_self==0 else 1]
        fret = lambda ws, x, *args: self._nn(ws, x*mask+offset, *args)*faux(x[...,axis_other])[...,None]
        
        return fret
        

    def interface_function(self, axis_self: Tuple[int], end_self: Tuple[int], axis_other: Tuple[int], end_other: Tuple[int], bounds_other: Tuple[Tuple[float]], decay_fun: callable=lambda x: x) -> callable:
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
        assert isinstance(axis_self, tuple) and all([isinstance(i, int) and i>=0 and i<self._d for i in axis_self]), "Check the axis_self argument."
        assert isinstance(end_self, tuple) and all([isinstance(i, int) and (i==0 or i==-1) for i in end_self]), "Check the end_self argument."
        assert isinstance(axis_other, tuple) and all([isinstance(i, int) and i>=0 and i<self._d for i in axis_other]), "Check the axis_other argument."
        assert isinstance(end_other, tuple) and all([isinstance(i, int) and (i==0 or i==-1) for i in end_other]), "Check the end_other argument."
         
        endpositive = np.array([bounds_other[a][e] for a,e in zip(axis_other, end_other)])
        endzero = np.array([bounds_other[a][-1 if e == 0 else 0] for a, e in zip(axis_other, end_other)])
        faux = lambda x: decay_fun((x-endzero)/(endpositive-endzero))
        
        mask = np.ones((self._d,))
        mask[list(axis_self)] = 0
        offset = np.zeros((self._d,))
        offset[list(axis_self)] = np.array([self._bounds[a][0 if e==0 else 1] for a, e in zip(axis_self, end_self)])
        fret = lambda ws, x, *args: self._nn(ws, x*mask+offset, *args)*jnp.prod(faux(x[...,list(axis_other)]), axis=-1)[...,None]
        
        return fret
        
class PINN():
    
    def __init__(self):
        self.neural_networks = {}
        self.neural_networks_initializers = {}
        self.weights = {}
        pass
    
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
         
    def loss(self,w):
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


def monom(x: jax.numpy.array, d: int, bound: float, deg: float = 1.0, out_dims: int = 1):
    return jnp.tile(((x[...,d]-bound)**deg)[...,None],out_dims)

def DirichletMask(out_dims, domain: list[tuple[float, float]], conditions: list[dict], alpha = 1.0):
  """Layer constructor function for a dense (fully-connected) layer."""
  def init_fun(rng, input_shape):
    output_shape = input_shape
    return output_shape, tuple()
  def apply_fun(params, inputs, **kwargs):
    res = 1
    for c in conditions:
        res = res * ((inputs[..., c['dim']]-(domain[c['dim']][0] if c['end'] == 0 else domain[c['dim']][1]))**alpha)[...,None]
    return jnp.tile(res, out_dims)
  return init_fun, apply_fun


        
        
def face_function(axis_this: int, end_this: int, axis_other: int, end_other: int, bounds: tuple[tuple[int]], decay_fun: callable, shape_out: tuple[int]=(1,)) -> callable:
    
    pass
        

def edge_function(dim: int, decay_fun: callable = lambda x: x, shape_out: tuple[int] =(1,)) -> callable:
    
    def func():
        pass
    
    return func 
