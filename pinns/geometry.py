import jax
import jax.numpy as jnp
import numpy as np

def tangent2normal_2d(tangents):
    rotation = np.array([[0,-1],[1,0]])
    return np.einsum('ij,mnj->mni',rotation,tangents)




class Patch():

    def __init__(self, rand_key) -> None:
        self.rand_key = rand_key

    def sample_inside(self, N, pdf = None):
        pass

    def sample_boundary(self, d, end, N, tangent = False, normal = False, pdf = None):
        pass
    def __call__(self,y):
        pass

class PatchParametrized(Patch):

    def __init__(self, parametrization, dims, rand_key):
        super(PatchParametrized, self).__init__(rand_key)
        self.dims = dims
        self._parametrization = parametrization
        self._parametrization_in = jax.jit(parametrization)
        self._parametrization_bd = jax.jit(parametrization)
        self._parametrization_bd_jacobian = jax.jit(jax.jacfwd(parametrization))

    def sample_inside(self, N, pdf=None):

        if pdf is None:
            ys = jax.random.uniform(self.rand_key, (N, self.dims))
            xs = self._parametrization_in(ys)

        return xs

    def sample_boundary(self, d, end, N, tangent = False, normal = False, pdf=None):
        
        if pdf is None:
            ys = jax.random.uniform(self.rand_key, (N, self.dims))
            ys = ys.at[:,d].set(end)
            xs = self._parametrization_bd(ys)

        return xs


class PatchNURBS(Patch):

    def __init__(self, basis, knots, weights, rand_key):
        super(PatchNURBS, self).__init__(rand_key)
        self.d = len(basis)
        self.basis = basis
        self.knots = knots 
        self.weights = weights

    def __call__(self,y):
          
        Bs = [b(y[:,i]) for i,b in enumerate(self.basis)]

        den = np.einsum('im,i...->m...',Bs[0],self.weights)
        for i in range(1,self.d):
            den = np.einsum('im,mi...->m...',Bs[i],den)

        xs = np.einsum('im,i...->m...',Bs[0],np.einsum('...i,...->...i',self.knots,self.weights))
        for i in range(1,self.d):
            xs = np.einsum('im,mi...->m...',Bs[i],xs)

        return np.einsum('...i,...->...i',xs,1/den)

    def _eval_derivative(self, y, dim):
        Bs = [b(y[:,i], derivative = False) for i,b in enumerate(self.basis)]
        dBs = [b(y[:,i], derivative = (dim ==i)) for i,b in enumerate(self.basis)]

        den = np.einsum('im,i...->m...',Bs[0],self.weights)
        for i in range(1,self.d):
            den = np.einsum('im,mi...->m...',Bs[i],den)
        den = np.tile(den[...,None],self.d)

        Dden = np.einsum('im,i...->m...',dBs[0],self.weights)
        for i in range(1,self.d):
            Dden = np.einsum('im,mi...->m...',dBs[i],Dden)
        Dden = np.tile(Dden[...,None],self.d)

        xs = np.einsum('im,i...->m...',Bs[0],np.einsum('...i,...->...i',self.knots,self.weights))
        for i in range(1,self.d):
            xs = np.einsum('im,mi...->m...',Bs[i],xs)
        
        Dxs = np.einsum('im,i...->m...',dBs[0],np.einsum('...i,...->...i',self.knots,self.weights))
        for i in range(1,self.d):
            Dxs = np.einsum('im,mi...->m...',dBs[i],Dxs)

        return (Dxs*den-xs*Dden)/(den**2)
    
    def _eval_omega(self,y):
        lst = []
        for d in self.d:
            lst.append(self._eval_derivative(y,d)[:,None,:])

        return np.concatenate(tuple(lst),1)

    def importance_sampling_2d(self, N, pdf = None):
        
        if pdf is None:
            pdf = lambda x: 1.0

        ys = np.random.rand(N,self.d)

        Gys = self.__call__(ys)

        

    def sample_inside(self, N, pdf=None):
        
        if pdf is None:
            ys = np.random.rand(N, self.d)
            # ys = jax.random.uniform(self.rand_key, (N, self.d))
            xs = self.__call__(ys)
        else:
            pass
            # Gy = self.__call__(ys)
        return xs

    def sample_boundary(self, d, end, N, tangent=False, normal=False, pdf=None):
        
        if pdf == None:
            y = np.random.rand(N,self.d)
            y[:,d] = end
            Bs = [b(y[:,i]) for i,b in enumerate(self.basis)]
            dBs = [b(y[:,i], derivative  = True) for i,b in enumerate(self.basis)]

            pts = self.__call__(y)
            pts_tangent = []
            for i in range(self.d):
                if i!=d:
                    ds = [False]*self.d
                    ds[i] = True
                    
                    v = self._eval_derivative(y, i)
                    v = v/np.tile(np.linalg.norm(v,axis = 1, keepdims=True),self.d)
                    pts_tangent += [v[:,None,:]]
            #v = self._eval_derivative(y, d)
            #norm = v/np.tile(np.linalg.norm(v,axis = 1, keepdims=True),self.d)
            return pts, np.concatenate(tuple(pts_tangent),1)#, norm

             


