import jax
import jax.numpy as jnp
import numpy as np
from .bspline import BSplineBasisJAX
from typing import Union, Callable, TypeVar, Generic, Any

def tangent2normal_2d(tangents):
    rotation = np.array([[0,-1],[1,0]])
    return np.einsum('ij,mnj->mni',rotation,tangents)

def tangent2normal_3d(tangents):

    result1 = tangents[...,1,0]*tangents[...,2,1]-tangents[...,2,0]*tangents[...,1,1]
    result2 = tangents[...,2,0]*tangents[...,0,1]-tangents[...,0,0]*tangents[...,2,1]
    result3 = tangents[...,0,0]*tangents[...,1,1]-tangents[...,1,0]*tangents[...,0,1]
    return np.concatenate((result1[...,None], result2[...,None], result3[...,None]),-1)

def normalize(vectors):
    norms = np.linalg.norm(vectors,axis=-1)
    return vectors / np.tile(norms[...,None],vectors.shape[-1])
    
def tensor_product_integration(bases, N):
    Ks = [b.quadrature_points(n)[0] for b,n in zip(bases,N)]
    Ws = [b.quadrature_points(n)[1] for b,n in zip(bases,N)]
    Knots = np.meshgrid(*Ks)
    points = np.concatenate(tuple([k.flatten()[:,None] for k in Knots]),-1)
    weights = np.ones((1,))
    for w in Ws:
        weights = np.kron(w,weights)
    return points, weights
    
class AffineTransformation():
    def __init__(self, Mat, offset):
        self.Mat = Mat
        self.offset = offset
    
    def __call__(self,points):
        return np.einsum('...j,ij->...i',points, self.Mat)+self.offset
    
    def metric_coefficients(self):
        pass    
    
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

    def __init__(self, basis, knots, weights, rand_key, bounds = None):
        super(PatchNURBS, self).__init__(rand_key)
        self.d = len(basis)
        self.basis = basis
        self.knots = knots 
        self.weights = weights
        self.dembedding = self.knots.shape[-1]
        if bounds == None:
            self.bounds = [(b.knots[0],b.knots[-1]) for b in self.basis]
        else:
            self.bounds = bounds
            
    def __call__(self, y, differential=False):
          
        Bs = [b(y[:,i]) for i,b in enumerate(self.basis)]

        den = np.einsum('im,i...->m...',Bs[0],self.weights)
        for i in range(1,self.d):
            den = np.einsum('im,mi...->m...',Bs[i],den)

        xs = np.einsum('im,i...->m...',Bs[0],np.einsum('...i,...->...i',self.knots,self.weights))
        for i in range(1,self.d):
            xs = np.einsum('im,mi...->m...',Bs[i],xs)

        Gys = np.einsum('...i,...->...i',xs,1/den)
        
        if differential:
            DGys = self._eval_omega(y)
            Weights = 1
            if self.d == 3 and self.dembedding == 3:
                diff = np.abs(DGys[:,0,0]*DGys[:,1,1]*DGys[:,2,2] + DGys[:,0,1]*DGys[:,1,2]*DGys[:,2,0]+DGys[:,0,2]*DGys[:,1,0]*DGys[:,2,1] - DGys[:,0,2]*DGys[:,1,1]*DGys[:,2,0] - DGys[:,0,0]*DGys[:,1,2]*DGys[:,2,1] - DGys[:,0,1]*DGys[:,1,0]*DGys[:,2,2] )*Weights  # type: ignore
            elif self.d==2 and self.dembedding == 2:
                diff = np.abs(DGys[:,0,0]*DGys[:,1,1] -  DGys[:,0,1]*DGys[:,1,0])*Weights
            elif self.d==2 and self.dembedding==3:
                diff = np.einsum('ij,i->ij',tangent2normal_3d(DGys),Weights)
            elif self.d==1:
                diff = DGys[...,:,0]*Weights
            else:
                diff = np.einsum('ij,i->ij',DGys,Weights)
            
        if differential:
            return Gys, diff   
        else: return Gys
    
         

    def __repr__(self):
        if self.d == 1:
            s = 'NURBS curve'
        elif self.d == 2:
            s = 'NURBS surface'
        elif self.d == 3:
            s = 'NURBS volume'
        else: 
            s = 'NURBS instance'
        
        s += ' embedded in a '+str(self.dembedding)+'D space.\n'
        s += 'Basis:\n' 
        for b in self.basis:
            s+=str(b)+'\n'
        
        return s
        
    def __getitem__(self, key):
        
        if len(key) != self.d:
            raise Exception('Invalid number of dimensions.')
        
        basis_new = []
        bounds_new = []
        knots = self.knots.copy()
        weights = self.weights.copy()
        

            
        axes = [] 
        for k, id in enumerate(key):
            if isinstance(id,int) or isinstance(id,float):
                if self.bounds[k][0]<=id and id<=self.bounds[k][1]:
                    axes.append(k)
                    
                    B = self.basis[k](id).flatten()
                    s = tuple([None]*k+[slice(None,None,None)]+[None]*(self.d-k-1))
                    B = B[s]
                    weights = weights*B
                    
                else:
                    raise Exception("Value must be inside the domain of the BSpline basis.")
            elif isinstance(id,slice):
                basis_new.append(self.basis[k])
                start = id.start if id.start!=None else self.bounds[k][0]
                stop = id.stop if id.stop!=None else self.bounds[k][1]
                bounds_new.append((start,stop))
            else:
                raise Exception("Only slices and scalars are permitted")

        weights_new = np.sum(weights,axis=tuple(axes))
        knots_new = np.sum(self.knots*weights[...,None], axis=tuple(axes))
        knots_new = knots_new/weights_new[...,None]
        
               
        return PatchNURBS(basis_new,knots_new, weights_new, self.rand_key, bounds=bounds_new)
        
    def _eval_derivative(self, y, dim):
        
        Bs = []
        dBs =[]
        
        for i in range(len(self.basis)):
            Bs.append(self.basis[i](y[...,i]))
            dBs.append(self.basis[i](y[:,i], derivative = (dim ==i)))

        den = jnp.einsum('im,i...->m...',Bs[0],self.weights)
        for i in range(1,self.d):
            den = jnp.einsum('im,mi...->m...',Bs[i],den)
        den = jnp.tile(den[...,None],self.dembedding)

        Dden = jnp.einsum('im,i...->m...',dBs[0],self.weights)
        for i in range(1,self.d):
            Dden = jnp.einsum('im,mi...->m...',dBs[i],Dden)
        Dden = jnp.tile(Dden[...,None],self.dembedding)

        xs = jnp.einsum('im,i...->m...',Bs[0],jnp.einsum('...i,...->...i',self.knots,self.weights))
        for i in range(1,self.d):
            xs = jnp.einsum('im,mi...->m...',Bs[i],xs)
        
        Dxs = jnp.einsum('im,i...->m...',dBs[0],jnp.einsum('...i,...->...i',self.knots,self.weights))
        for i in range(1,self.d):
            Dxs = jnp.einsum('im,mi...->m...',dBs[i],Dxs)

        return (Dxs*den-xs*Dden)/(den**2)
    
    def _eval_omega(self,y):
        """
        Evaluate the derivative of the parametrization (jacobian).
        y[...,i,j] = \partial_y
        
        Args:
            y (numpy.array): the points

        Returns:
            numpy.array: _description_
        """
        lst = []
        for d in range(self.d):
            lst.append(self._eval_derivative(y,d)[:,:,None])

        return jnp.concatenate(tuple(lst),-1)

    def importance_sampling(self, N, pdf = None):
        
        if pdf is None:
            pdf = lambda x: 1.0
            
        vol_ref = 1.0
        for i  in self.bounds:
            vol_ref *= i[1]-i[0]
            
        ys = np.random.rand(N,self.d)*np.array([i[1]-i[0] for i in self.bounds]) + np.array([i[0] for i in self.bounds])       
        Gys = self.__call__(ys)
        
        DGys = self._eval_omega(ys)
        
        if self.d == 3 and self.dembedding == 3:
            diff = np.abs(DGys[:,0,0]*DGys[:,1,1]*DGys[:,2,2] + DGys[:,0,1]*DGys[:,1,2]*DGys[:,2,0]+DGys[:,0,2]*DGys[:,1,0]*DGys[:,2,1] - DGys[:,0,2]*DGys[:,1,1]*DGys[:,2,0] - DGys[:,0,0]*DGys[:,1,2]*DGys[:,2,1] - DGys[:,0,1]*DGys[:,1,0]*DGys[:,2,2] )
        elif self.d==2 and self.dembedding == 2:
            diff = np.abs(DGys[:,0,0]*DGys[:,1,1] -  DGys[:,0,1]*DGys[:,1,0])
        elif self.d==2 and self.dembedding==3:
            diff = tangent2normal_3d(DGys)
        elif self.d==1:
            diff = DGys[...,:,0] 
        else:
            diff = DGys
        return Gys, diff*vol_ref/N
     
    def quadrature(self, N = 32, knots = 'leg'):
        
       
       
        
        Knots = [(np.polynomial.legendre.leggauss(N)[0]+1)*0.5*(self.bounds[i][1]-self.bounds[i][0])+self.bounds[i][0] for i in range(self.d)]
        Ws = [np.polynomial.legendre.leggauss(N)[1]*0.5*(self.bounds[i][1]-self.bounds[i][0]) for i in range(self.d)]
        Knots = np.meshgrid(*Knots)
        ys = np.concatenate(tuple([k.flatten()[:,None] for k in Knots]),-1)
        
        Weights = Ws[0]
        for i in range(1,self.d):
            Weights = np.kron(Weights, Ws[i])
                   
        Gys = self.__call__(ys)
        
        DGys = self._eval_omega(ys)
        
        if self.d == 3 and self.dembedding == 3:
            diff = np.abs(DGys[:,0,0]*DGys[:,1,1]*DGys[:,2,2] + DGys[:,0,1]*DGys[:,1,2]*DGys[:,2,0]+DGys[:,0,2]*DGys[:,1,0]*DGys[:,2,1] - DGys[:,0,2]*DGys[:,1,1]*DGys[:,2,0] - DGys[:,0,0]*DGys[:,1,2]*DGys[:,2,1] - DGys[:,0,1]*DGys[:,1,0]*DGys[:,2,2] )*Weights
        elif self.d==2 and self.dembedding == 2:
            diff = np.abs(DGys[:,0,0]*DGys[:,1,1] -  DGys[:,0,1]*DGys[:,1,0])*Weights
        elif self.d==2 and self.dembedding==3:
            diff = np.einsum('ij,i->ij',tangent2normal_3d(DGys),Weights)
        elif self.d==1:
            diff = DGys[...,:,0]*Weights
        else:
            diff = np.einsum('ij,i->ij',DGys,Weights)
        return Gys, diff   
    
    def importance_sampling_2d(self, N, pdf = None):
        
        if pdf is None:
            pdf = lambda x: 1.0

        ys = np.random.rand(N,self.d)

        Gys = self.__call__(ys)

        DGys = self._eval_omega(ys)
         
        det = np.abs(DGys[:,0,0]*DGys[:,1,1] -  DGys[:,0,1]*DGys[:,1,0])
        
        return Gys, det
    
    

    def importance_sampling_3d(self, N, pdf = None, bounds = None):
        
        if pdf is None:
            pdf = lambda x: 1.0
        if bounds ==None:
            bounds = ((0,1),(0,1),(0,1))
            
        ys = np.random.rand(N,self.d)*np.array([bounds[0][1]-bounds[0][0],bounds[1][1]-bounds[1][0],bounds[2][1]-bounds[2][0]]) + np.array([bounds[0][0],bounds[1][0],bounds[2][0]])       
        Gys = self.__call__(ys)

        DGys = self._eval_omega(ys)
         
        det = np.abs(DGys[:,0,0]*DGys[:,1,1]*DGys[:,2,2] + DGys[:,0,1]*DGys[:,1,2]*DGys[:,2,0]+DGys[:,0,2]*DGys[:,1,0]*DGys[:,2,1] - DGys[:,0,2]*DGys[:,1,1]*DGys[:,2,0] - DGys[:,0,0]*DGys[:,1,2]*DGys[:,2,1] - DGys[:,0,1]*DGys[:,1,0]*DGys[:,2,2] )
        det = det/det.size*(bounds[0][1]-bounds[0][0])*(bounds[1][1]-bounds[1][0])*(bounds[2][1]-bounds[2][0])
        

            
        return Gys, det
    
    def sample_inside(self, N, pdf=None):
        
        if pdf is None:
            ys = np.random.rand(N, self.d)
            # ys = jax.random.uniform(self.rand_key, (N, self.d))
            xs = self.__call__(ys)
        else:
            pass
            # Gy = self.__call__(ys)
        return xs
    
    
    
    def sample_boundary(self, d, end, N, normalize=True, pdf=None):
        
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
                    if normalize:
                        v = v/np.tile(np.linalg.norm(v,axis = 1, keepdims=True),self.d)
                    pts_tangent += [v[:,None,:]]
            #v = self._eval_derivative(y, d)
            #norm = v/np.tile(np.linalg.norm(v,axis = 1, keepdims=True),self.d)
            return pts, np.concatenate(tuple(pts_tangent),1)#, norm

             

Array = np.ndarray | jnp.DeviceArray

class PatchNURBSParam(Patch):
    __d : int
    __dembedding : int
    __dparam: int
    __basis : list[BSplineBasisJAX]
    __weights: Callable | Array
    __knots: Callable | Array
    __bounds: list[tuple[float,float]]
    __N: list[int]
    
    @property
    def d(self):
        return self.__d
    
    def knots(self, params: Array|None = None) -> Array:
        if params is None:
            return self.__knots
        else: 
            return self.__knots(params)
    
    def weights(self, params: Array|None = None) -> Array:
        if params is None:
            return self.__weights
        else: 
            return self.__weights(params)
    
    def __init__(self, basis : list[BSplineBasisJAX], knots : Array | Callable, weights: Array | Callable, dparam : int, dembedding :int, rand_key: jax.random.PRNGKey, bounds = None):
        super(PatchNURBSParam, self).__init__(rand_key)
        
        if not ((dparam>0 and callable(weights) and callable(knots)) or (dparam==0 and not callable(knots) and not callable(weights))):
            raise Exception("The weights and knots are callable iff the number of parameters is greater than 0.")
        
        self.__d = len(basis)
        self.__basis = basis
        self.__knots = knots 
        self.__weights = weights
        self.__dembedding = dembedding
        self.__dparam = dparam
        self.__N = [b.n for b in basis]
        
        if bounds == None:
            self.__bounds = [(b.knots[0],b.knots[-1]) for b in basis]
        else:
            self.__bounds = bounds
            
    def __call__(self, y: Array, params: Array|None = None, differential: bool =False) -> Array | tuple[Array, Array]:
          
        Bs = [b(y[:,i]).T for i,b in enumerate(self.__basis)]

        if self.__dparam == 0:
            den = jnp.einsum('mi,i...->m...',Bs[0],self.__weights)
            for i in range(1,self.__d):
                den = jnp.einsum('mi,mi...->m...',Bs[i],den)
        else:
            def tmp(w, *bs):  # type: ignore
                den = jnp.einsum('i,i...->...',bs[0],w)
                for i in range(1,self.__d):
                    den = jnp.einsum('i,i...->...',bs[i],den)
                
                return den
            den = jax.vmap(lambda x , *args: tmp(self.__weights(x), *args),(0))(params, *Bs)
            
                
        if self.__dparam == 0:
            xs = jnp.einsum('mi,i...->m...',Bs[0],jnp.einsum('...i,...->...i',self.__knots,self.__weights))
            for i in range(1,self.__d):
                xs = jnp.einsum('mi,mi...->m...',Bs[i],xs)
        else:
            def tmp(w,k, *bs):
                xs = jnp.einsum('i,i...->...',bs[0],jnp.einsum('...i,...->...i',k,w))
                for i in range(1,self.__d):
                    xs = jnp.einsum('i,i...->...',bs[i],xs)
                return xs
                    
            xs = jax.vmap(lambda x, *args: tmp(self.__weights(x), self.__knots(x), *args))(params,*Bs)
        Gys = jnp.einsum('...i,...->...i',xs,1/den)
        
        if differential:
            DGys = self._eval_omega(y, params)
            Weights = 1
            if self.__d == 3 and self.__dembedding == 3:
                diff = jnp.abs(DGys[:,0,0]*DGys[:,1,1]*DGys[:,2,2] + DGys[:,0,1]*DGys[:,1,2]*DGys[:,2,0]+DGys[:,0,2]*DGys[:,1,0]*DGys[:,2,1] - DGys[:,0,2]*DGys[:,1,1]*DGys[:,2,0] - DGys[:,0,0]*DGys[:,1,2]*DGys[:,2,1] - DGys[:,0,1]*DGys[:,1,0]*DGys[:,2,2] )*Weights
            elif self.__d==2 and self.__dembedding == 2:
                diff = jnp.abs(DGys[:,0,0]*DGys[:,1,1] -  DGys[:,0,1]*DGys[:,1,0])*Weights
            elif self.__d==2 and self.__dembedding==3:
                def tmp(tangents):
                    result1 = tangents[...,1,0]*tangents[...,2,1]-tangents[...,2,0]*tangents[...,1,1]
                    result2 = tangents[...,2,0]*tangents[...,0,1]-tangents[...,0,0]*tangents[...,2,1]
                    result3 = tangents[...,0,0]*tangents[...,1,1]-tangents[...,1,0]*tangents[...,0,1]
                    return jnp.concatenate((result1[...,None], result2[...,None], result3[...,None]),-1)
                diff = tmp(DGys) # jnp.einsum('ij,i->ij',tmp(DGys),Weights)
            elif self.__d==1:
                diff = DGys[...,:,0]*Weights
            else:
                diff = DGys # jnp.einsum('ij,i->ij',DGys,Weights)
            
        if differential:
            return Gys, diff   
        else: return Gys
    
         

    def __repr__(self):
        if self.__d == 1:
            s = 'NURBS curve'
        elif self.__d == 2:
            s = 'NURBS surface'
        elif self.__d == 3:
            s = 'NURBS volume'
        else: 
            s = 'NURBS instance'
        
        s += ' embedded in a '+str(self.__dembedding)+'D space depending on ' + str(self.__dparam) + ' parameters.\n'
        s += 'Basis:\n' 
        for b in self.__basis:
            s+=str(b)+'\n'
        
        return s
        
    def __getitem__(self, key):
        
        if len(key) != self.__d+self.__dparam:
            raise Exception('Invalid number of dimensions. It must equal the number of spacial dimensions + number of parameters.')
        
        axes = []
        basis_new = []
        bounds_new = []
        
        if isinstance(key[-1], ellipsis):
            key = key[:-1]+[slice(None,None,None)]*self.__dparam
        
        weights_mult = jnp.ones(self.__N)
        transform = False
        for k in range(self.__d):
            id = key[k]
            if isinstance(id,int) or isinstance(id,float):
                if self.__bounds[k][0]<=id and id<=self.__bounds[k][1]:
                    axes.append(k)
                    B = self.__basis[k](np.array([id])).flatten()
                    s = tuple([None]*k+[slice(None,None,None)]+[None]*(self.__d-k-1))
                    B = B[s]
                    weights_mult = weights_mult*B
                    transform = True                    
                else:
                    raise Exception("Value must be inside the domain of the BSpline basis.")
            elif isinstance(id,slice):
                basis_new.append(self.__basis[k])
                start = id.start if id.start!=None else self.__bounds[k][0]
                stop = id.stop if id.stop!=None else self.__bounds[k][1]
                bounds_new.append((start,stop))
            else:
                raise Exception("Only slices, scalars and ellipsis are permitted")
            
        if self.__dparam == 0:
            weights_new = jnp.sum(self.__weights*weights_mult,axis=tuple(axes))
            knots_new = jnp.sum(self.__knots*(self.__weights*weights_mult)[...,None], axis=tuple(axes))
            knots_new = knots_new/weights_new[...,None]
            param_new = 0
        else:
            weights_new = lambda *args: jnp.sum(self.__weights(*args)*weights_mult, axis=tuple(axes))
            knots_new = lambda *args: jnp.sum(self.__knots(*args)*(self.__weights(*args)*weights_mult)[...,None], axis=tuple(axes))/weights_new(*args)[...,None]
               
            params_take = []
            vect = []
            
            for k in range(self.__dparam):
                id = key[k+self.__d]
                if isinstance(id,int) or isinstance(id,float):
                    vect.append(float(id))
                elif isinstance(id,slice) and (id.start == None and id.stop==None and id.step==None):
                    params_take.append(k)
                    vect.append(0.0)
                else:
                    raise Exception("Only slices, scalars and ellipsis are permitted")
                
            E = np.eye(self.__dparam)[tuple(params_take),:]
            vect = np.array(vect)
            
            if len(params_take)!=0:
                weights_new = lambda ps: jnp.sum(self.__weights(jnp.einsum('...i,ij->...j',ps,E)+vect)*weights_mult, axis=tuple(axes))
                knots_new = lambda ps: jnp.sum(self.__knots(jnp.einsum('...i,ij->...j',ps,E)+vect)*(self.__weights(jnp.einsum('...i,ij->...j',ps,E)+vect)*weights_mult)[...,None], axis=tuple(axes))/weights_new(jnp.einsum('...i,ij->...j',ps,E)+vect)[...,None]
            else:
                weights_new = jnp.sum(self.__weights(vect)*weights_mult, axis=tuple(axes))
                knots_new =  jnp.sum(self.__knots(vect)*(self.__weights(vect)*weights_mult)[...,None], axis=tuple(axes))/weights_new[...,None]
            param_new = len(params_take)
        
        return PatchNURBSParam(basis_new, knots_new, weights_new, param_new, self.__dembedding, self.rand_key, bounds=bounds_new)
        
    def _eval_derivative(self, y: Array, params: Array|None, dim: int)->Array:
        
        Bs = []
        dBs =[]
        
        for i in range(len(self.__basis)):
                Bs.append(self.__basis[i](y[...,i]).T)
                dBs.append(self.__basis[i](y[:,i], derivative = (dim ==i)).T)

        if self.__dparam == 0:
        
            den = jnp.einsum('mi,i...->m...',Bs[0],self.__weights)
            for i in range(1,self.__d):
                den = jnp.einsum('mi,mi...->m...',Bs[i],den)
            den = jnp.tile(den[...,None],self.__dembedding)

            Dden = jnp.einsum('mi,i...->m...',dBs[0],self.__weights)
            for i in range(1,self.__d):
                Dden = jnp.einsum('mi,mi...->m...',dBs[i],Dden)
            Dden = jnp.tile(Dden[...,None],self.__dembedding)

            xs = jnp.einsum('mi,i...->m...',Bs[0],jnp.einsum('...i,...->...i',self.__knots,self.__weights))
            for i in range(1,self.__d):
                xs = jnp.einsum('mi,mi...->m...',Bs[i],xs)
            
            Dxs = jnp.einsum('mi,i...->m...',dBs[0],jnp.einsum('...i,...->...i',self.__knots,self.__weights))
            for i in range(1,self.__d):
                Dxs = jnp.einsum('mi,mi...->m...',dBs[i],Dxs)
        else:
            def tmp(w, *bs):
                den = jnp.einsum('i,i...->...',bs[0],w)
                for i in range(1,self.__d):
                    den = jnp.einsum('i,i...->...',bs[i],den)
                den = jnp.tile(den[...,None],self.__dembedding)
                return den
            
            tmp_jit = jax.vmap(lambda x , *args: tmp(self.__weights(x), *args))
            den = tmp_jit(params, *Bs)
            Dden = tmp_jit(params, *dBs)
            
            def tmp(w,k, *bs):
                xs = jnp.einsum('i,i...->...',bs[0],jnp.einsum('...i,...->...i',k,w))
                for i in range(1,self.__d):
                    xs = jnp.einsum('i,i...->...',bs[i],xs)
                return xs
            
            tmp_jit = jax.vmap(lambda x, *args: tmp(self.__weights(x), self.__knots(x), *args)) 
            xs = tmp_jit(params,*Bs)
            Dxs = tmp_jit(params,*dBs)
            
        return (Dxs*den-xs*Dden)/(den**2)
    
    def GetJacobian(self, y: Array, params: Array | None = None) -> jax.numpy.DeviceArray:
        """
        Evaluate the Jacobian of the geometry transformation

        Args:
            y (jax.numpy.array): the positions in the reference domain. Has the shape N x d.

        Returns:
            jax.numpy.array: _description_
        """
        
        lst = []
        for d in range(self.__d):
            lst.append(self._eval_derivative(y,params, d)[:,:,None])

        return jnp.concatenate(tuple(lst),-1)
        
    def GetMetricTensors(self, y: Array, params: Array|None = None) -> tuple[jnp.DeviceArray,jnp.DeviceArray,jnp.DeviceArray]:
        
        
        DGys = self.GetJacobian(y, params)
        Inv = jnp.linalg.inv(DGys)
        omega = jnp.abs(jnp.linalg.det(DGys))
        K = jnp.einsum('mij,mjk,m->mik',Inv,jnp.transpose(Inv,[0,2,1]),omega)
        Gr = Inv

        return (omega,Gr,K)
    
    def _eval_omega(self,y: Array, params:Array|None = None):
        """
        Evaluate the derivative of the parametrization (jacobian).
        y[...,i,j] = \partial_y
        
        Args:
            y (numpy.array): the points

        Returns:
            numpy.array: _description_
        """
        lst = []
        for d in range(self.__d):
            lst.append(self._eval_derivative(y, params, d)[:,:,None])

        return jnp.concatenate(tuple(lst),-1)

    
    def importance_sampling(self, N, pdf = None):
        
        if pdf is None:
            pdf = lambda x: 1.0
            
        vol_ref = 1.0
        for i  in self.__bounds:
            vol_ref *= i[1]-i[0]
            
        ys = np.random.rand(N,self.__d)*np.array([i[1]-i[0] for i in self.__bounds]) + np.array([i[0] for i in self.__bounds])       
        Gys = self.__call__(ys)
        
        DGys = self._eval_omega(ys)
        
        if self.__d == 3 and self.__dembedding == 3:
            diff = np.abs(DGys[:,0,0]*DGys[:,1,1]*DGys[:,2,2] + DGys[:,0,1]*DGys[:,1,2]*DGys[:,2,0]+DGys[:,0,2]*DGys[:,1,0]*DGys[:,2,1] - DGys[:,0,2]*DGys[:,1,1]*DGys[:,2,0] - DGys[:,0,0]*DGys[:,1,2]*DGys[:,2,1] - DGys[:,0,1]*DGys[:,1,0]*DGys[:,2,2] )
        elif self.__d==2 and self.__dembedding == 2:
            diff = np.abs(DGys[:,0,0]*DGys[:,1,1] -  DGys[:,0,1]*DGys[:,1,0])
        elif self.__d==2 and self.__dembedding==3:
            diff = tangent2normal_3d(DGys)
        elif self.__d==1:
            diff = DGys[...,:,0] 
        else:
            diff = DGys
        return Gys, diff*vol_ref/N
     
    def quadrature(self, N = 32, knots = 'leg'):
        
       
       
        
        Knots = [(np.polynomial.legendre.leggauss(N)[0]+1)*0.5*(self.bounds[i][1]-self.bounds[i][0])+self.bounds[i][0] for i in range(self.d)]
        Ws = [np.polynomial.legendre.leggauss(N)[1]*0.5*(self.bounds[i][1]-self.bounds[i][0]) for i in range(self.d)]
        Knots = np.meshgrid(*Knots)
        ys = np.concatenate(tuple([k.flatten()[:,None] for k in Knots]),-1)
        
        Weights = Ws[0]
        for i in range(1,self.d):
            Weights = np.kron(Weights, Ws[i])
                   
        Gys = self.__call__(ys)
        
        DGys = self._eval_omega(ys)
        
        if self.d == 3 and self.dembedding == 3:
            diff = np.abs(DGys[:,0,0]*DGys[:,1,1]*DGys[:,2,2] + DGys[:,0,1]*DGys[:,1,2]*DGys[:,2,0]+DGys[:,0,2]*DGys[:,1,0]*DGys[:,2,1] - DGys[:,0,2]*DGys[:,1,1]*DGys[:,2,0] - DGys[:,0,0]*DGys[:,1,2]*DGys[:,2,1] - DGys[:,0,1]*DGys[:,1,0]*DGys[:,2,2] )*Weights
        elif self.d==2 and self.dembedding == 2:
            diff = np.abs(DGys[:,0,0]*DGys[:,1,1] -  DGys[:,0,1]*DGys[:,1,0])*Weights
        elif self.d==2 and self.dembedding==3:
            diff = np.einsum('ij,i->ij',tangent2normal_3d(DGys),Weights)
        elif self.d==1:
            diff = DGys[...,:,0]*Weights
        else:
            diff = np.einsum('ij,i->ij',DGys,Weights)
        return Gys, diff   
    
    def importance_sampling_2d(self, N, pdf = None):
        
        if pdf is None:
            pdf = lambda x: 1.0

        ys = np.random.rand(N,self.d)

        Gys = self.__call__(ys)

        DGys = self._eval_omega(ys)
         
        det = np.abs(DGys[:,0,0]*DGys[:,1,1] -  DGys[:,0,1]*DGys[:,1,0])
        
        return Gys, det
    
    

    def importance_sampling_3d(self, N, pdf = None, bounds = None):
        
        if pdf is None:
            pdf = lambda x: 1.0
        if bounds ==None:
            bounds = ((0,1),(0,1),(0,1))
            
        ys = np.random.rand(N,self.d)*np.array([bounds[0][1]-bounds[0][0],bounds[1][1]-bounds[1][0],bounds[2][1]-bounds[2][0]]) + np.array([bounds[0][0],bounds[1][0],bounds[2][0]])       
        Gys = self.__call__(ys)

        DGys = self._eval_omega(ys)
         
        det = np.abs(DGys[:,0,0]*DGys[:,1,1]*DGys[:,2,2] + DGys[:,0,1]*DGys[:,1,2]*DGys[:,2,0]+DGys[:,0,2]*DGys[:,1,0]*DGys[:,2,1] - DGys[:,0,2]*DGys[:,1,1]*DGys[:,2,0] - DGys[:,0,0]*DGys[:,1,2]*DGys[:,2,1] - DGys[:,0,1]*DGys[:,1,0]*DGys[:,2,2] )
        det = det/det.size*(bounds[0][1]-bounds[0][0])*(bounds[1][1]-bounds[1][0])*(bounds[2][1]-bounds[2][0])
        

            
        return Gys, det
    
    def sample_inside(self, N, pdf=None):
        
        if pdf is None:
            ys = np.random.rand(N, self.d)
            # ys = jax.random.uniform(self.rand_key, (N, self.d))
            xs = self.__call__(ys)
        else:
            pass
            # Gy = self.__call__(ys)
        return xs
    
    
    
    def sample_boundary(self, d, end, N, normalize=True, pdf=None):
        
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
                    if normalize:
                        v = v/np.tile(np.linalg.norm(v,axis = 1, keepdims=True),self.d)
                    pts_tangent += [v[:,None,:]]
            #v = self._eval_derivative(y, d)
            #norm = v/np.tile(np.linalg.norm(v,axis = 1, keepdims=True),self.d)
            return pts, np.concatenate(tuple(pts_tangent),1)#, norm