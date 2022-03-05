import unittest
import pinns.operators
import jax
import jax.numpy as jnp
import numpy as np

import jax.config
jax.config.update("jax_enable_x64", True)


class TestOperators(unittest.TestCase):

    def test_gradient(self):

        func_scalar = lambda x : 1/jnp.sum(x**2,-1)[...,None]
        grad_reference = lambda x: -2*x/jnp.tile(jnp.sum(x**2,-1)[...,None],x.shape[1])**2
       
        grad_computed = pinns.operators.gradient(func_scalar)
        x = jnp.array(np.random.rand(100,8))
        gc = grad_computed(x)
        gr = grad_reference(x)
        error = jnp.linalg.norm(gc-gr)
        self.assertLess(error.to_py(),1e-13,"pinns.operators.gradient error: wrong gradient.")


    def test_laplace(self):

        func_scalar = lambda x: (x[...,0]**2+x[...,1]**2+x[...,2]**2+jnp.sin(x[...,1]))[...,None]
        laplace_reference = lambda x: (6-jnp.sin(x[:,1]))[...,None]

        laplace_computed = pinns.operators.laplace(func_scalar)
        x = jnp.array(np.random.rand(100,3))
        lr = laplace_reference(x)
        lc = laplace_computed(x)
        error = jnp.linalg.norm(lc-lr)
        self.assertLess(error.to_py(),1e-13,"pinns.operators.lapalce error: wrong laplace.")

    def test_divergence(self):

        func = lambda x: jnp.concatenate((jnp.sin(x[...,0])[...,None],jnp.exp(2*x[...,1])[...,None]), -1)
        divergence_reference = lambda x: (jnp.cos(x[...,0])+2*jnp.exp(2*x[...,1]))[...,None]

        divergence_computed = pinns.operators.divergence(func)
        x = jnp.array(np.random.rand(128,2))
        dr = divergence_reference(x)
        dc = divergence_computed(x)
        error = jnp.linalg.norm(dc-dr)
        self.assertLess(error.to_py(),1e-13,"pinns.operators.divergence error: wrong divergence.")

    def test_curl2d(self):

        func = lambda x: jnp.concatenate((jnp.sin(x[...,0])[...,None]*x[...,1],x[...,0]**2+jnp.exp(2*x[...,1])[...,None]), -1)
        curl2d_reference = lambda x: ( 2*x[...,0] - jnp.sin(x[...,0]) )[...,None]

        curl2d_computed = pinns.operators.curl2d(func)
        x = jnp.array(np.random.rand(128,2))
        cr = curl2d_reference(x)
        cc = curl2d_computed(x)
        error = jnp.linalg.norm(cc-cr)
        self.assertLess(error.to_py(),1e-13,"pinns.operators.curl2d error: wrong curl.")

    def test_curl3d(self):
        
        func = lambda x: jnp.concatenate( ( (x[...,0]+x[...,1]**2)[...,None] , (x[...,1]+x[...,2]**2)[...,None] , (x[...,2]+x[...,0]**2)[...,None] ) , -1)
        curl3d_reference = lambda x: np.concatenate( (-2*x[...,2][...,None],-2*x[...,0][...,None],-2*x[...,1][...,None]) ,-1)
        
        curl3d_computed = pinns.operators.curl3d(func)
        x = jnp.array(np.random.rand(128,3))
        cr = curl3d_reference(x)
        cc = curl3d_computed(x)
        error = jnp.linalg.norm(cc-cr)
        self.assertLess(error.to_py(),1e-13,"pinns.operators.curl3d error: wrong curl.")
        
if __name__ == '__main__':
    unittest.main()