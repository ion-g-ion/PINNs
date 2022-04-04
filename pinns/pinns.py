import jax
import jax.numpy as jnp
import numpy as np
from jax.example_libraries import stax, optimizers
import matplotlib.pyplot as plt
import datetime
import jax.scipy.optimize
import jax.flatten_util

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

    def add_trainable_parameter(self, name, shape):

        self.weights[name] = jax.random.uniform(self.key, shape)

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


    def lossgrad_handle(self, w):
        ws = self.weights_unravel(w)
        
        l = self.loss(ws)
        gr = jax.grad(self.loss)(ws)
        
        gr,_ = jax.flatten_util.ravel_pytree(gr)
        return l, gr


        
        