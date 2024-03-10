from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from ..functions import BSplineBasisJAX, PiecewiseBernsteinBasisJAX, jacobian, UnivariateFunctionBasis
from typing import Union, Callable, TypeVar, Generic, Any, Tuple, List
from ._base import rotation_matrix_3d
from ._distances import distance_to_bezier_curve_simple, distance_to_bezier_surface_simple
import abc 


def tangent2normal_2d(tangents):
    rotation = np.array([[0, -1], [1, 0]])
    return np.einsum('ij,mnj->mni', rotation, tangents)


def tangent2normal_3d(tangents):

    result1 = tangents[..., 1, 0]*tangents[..., 2, 1] - \
        tangents[..., 2, 0]*tangents[..., 1, 1]
    result2 = tangents[..., 2, 0]*tangents[..., 0, 1] - \
        tangents[..., 0, 0]*tangents[..., 2, 1]
    result3 = tangents[..., 0, 0]*tangents[..., 1, 1] - \
        tangents[..., 1, 0]*tangents[..., 0, 1]
    return np.concatenate((result1[..., None], result2[..., None], result3[..., None]), -1)


def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=-1)
    return vectors / np.tile(norms[..., None], vectors.shape[-1])


def tensor_product_integration(bases, N):
    Ks = [b.quadrature_points(n)[0] for b, n in zip(bases, N)]
    Ws = [b.quadrature_points(n)[1] for b, n in zip(bases, N)]
    Knots = np.meshgrid(*Ks)
    points = np.concatenate(tuple([k.flatten()[:, None] for k in Knots]), -1)
    weights = np.ones((1,))
    for w in Ws:
        weights = np.kron(w, weights)
    return points, weights



           
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
        """
        

        Args:
            d (int): _description_
            de (int): _description_
            points_reference (jax.Array): _description_
            points (jax.Array): _description_
            jacobian (jax.Array): _description_
            weights (jax.Array): _description_
            parameters (jax.Array | None): _description_
            weights_parameters (jax.Array | None): _description_
            inverted (float, optional): _description_. Defaults to 1.0.
        """
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
        """
        The points in the reference domain.

        Returns:
            jax.Array: the points as a `(N,d)` matrix.
        """
        return self.__ys
    
    @property 
    def points_physical(self) -> jax.Array:
        """
        The points transformed in the physical domain.

        Returns:
            jax.Array: the points as a `(N,d)` matrix.
        """
        return self.__xs

    @property 
    def weights(self) -> jax.Array:
        """
        Integration weights.

        Returns:
            jax.Array: th weights as a `(N,)` array.
        """
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
            




Array = np.ndarray | jax.Array

class Patch(metaclass=abc.ABCMeta):

    __d: int
    __dembedding: int
    __dparam: int
    __bounds: list[tuple[float, float]]
    
    def __init__(self, d: int, dembedding: int, dparam: int, bounds: list[tuple[float, float]]):
        self.__d = d 
        self.__dembedding = dembedding 
        self.__dparam = dparam 
        self.__bounds = bounds 
        
    @classmethod
    def __subclasshook__(cls, subclass):
            return (hasattr(subclass, '__repr__') and 
                callable(subclass.__repr__) and 
                hasattr(subclass, '__call__') and 
                callable(subclass.__call__) and 
                hasattr(subclass, '__getitem__') and 
                callable(subclass.__getitem__) and 
                hasattr(subclass, 'rotate') and 
                callable(subclass.rotate) and 
                hasattr(subclass, 'translate') and 
                callable(subclass.translate) and 
                hasattr(subclass, 'affine_transformation') and 
                callable(subclass.affine_transformation) and 
                hasattr(subclass, 'copy') and 
                callable(subclass.copy) and 
                hasattr(subclass, 'd') and 
                isinstance(subclass.d, property) and 
                hasattr(subclass, 'dembedding') and 
                isinstance(subclass.dembedding, property) and 
                hasattr(subclass, 'domain') and 
                isinstance(subclass.domain, property) and 
                hasattr(subclass, 'dparam') and 
                isinstance(subclass.dparam, property) and 
                hasattr(subclass, 'importance_sampling') and 
                callable(subclass.importance_sampling)) or NotImplemented

    @abc.abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, y: Array, params: Array | None, derivative: bool = False) -> Array | Tuple[Array, Array]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def __getitem__(self, key) -> Patch:
        raise NotImplementedError

    @abc.abstractmethod
    def rotate(self, angles: Tuple[float]):
        raise NotImplementedError
    
    @abc.abstractmethod
    def translate(self, offset: Tuple[float]):
        raise NotImplementedError 
    
    @abc.abstractmethod
    def affine_transformation(self, A: np.ndarray, b: np.ndarray):
        raise NotImplementedError
    
    @abc.abstractmethod
    def copy(self) -> Patch:
        """ Copy """
        raise NotImplementedError
    
    
    @property
    def d(self) -> int:
        """
        The dimension of the manifold.

        Returns:
            int: the number of dimensions.
        """
        return self.__d

    @property 
    def dembedding(self) -> int:
        """
        Returns the number of dimensions where the patch lives.

        Returns:
            int: the number of dimensions.
        """
        return self.__dembedding
    
    @property
    def domain(self) -> list[tuple[float, float]]:
        """
        The definition domain of the patch. Without parameters.

        Returns:
            list[tuple[float, float]]: tuple of intervals.
        """
        return self.__bounds

    @property 
    def dparam(self) -> int:
        """
        Return the number of parameters.

        Returns:
            int: number of parameters
        """
        return self.__dparam
    
    def importance_sampling(self, N: int, key: jax.random.PRNGKey, bounds_integration: list[tuple[float, float]] | None, bounds_parameters: list[tuple[float, float]] | None, pdf: Callable | None, inverted: float = 1.0) -> IsogeometricForm:
        
        
        if not (self.__dparam == 0 or bounds_parameters is not None):
            raise ValueError("If parameters are available, the bounds must be given.")
        
        
        if bounds_integration is None:
            bounds_integration = self.__bounds.copy()
            
        selector = [i for i in range(self.d) if bounds_integration[i][0] < bounds_integration[i][1]]
        axes = [i for i in range(self.d) if bounds_integration[i][0] == bounds_integration[i][1]]
        
        bounds_integration = np.array(tuple(bounds_integration))
        
        ys = jax.random.uniform(key ,(N,self.d))*(bounds_integration[:, 1]-bounds_integration[:, 0]) + bounds_integration[:, 0]
        Weights = jnp.ones((N,))*np.prod(bounds_integration[selector,1]-bounds_integration[selector,0])/ys.shape[0]
        
        if self.__dparam != 0:
            bounds_parameters = np.array(tuple(bounds_parameters))
            ps = jax.random.uniform(key ,(N,self.d))*(bounds_parameters[:, 1]-bounds_parameters[:, 0]) + bounds_parameters[:, 0]
            Weights_ps = jnp.ones((N,))*np.prod(bounds_parameters[selector,1]-bounds_parameters[selector,0])/ps.shape[0]
        else:
            ps = None 
            Weights_ps = None
        DG = self.__call__(ys, ps if self.__dparam != 0 else None, True)[...,:,selector]
        xs = self.__call__(ys, ps if self.__dparam != 0 else None, True)
        
        return IsogeometricForm(self.__d-len(axes), self.__dembedding, ys, xs, DG, Weights, ps, Weights_ps, inverted)

class PatchNURBS(Patch):
    __basis: list[BSplineBasisJAX]
    __weights: Callable | Array
    __knots: Callable | Array
    __N: list[int]
    
    @property
    def basis(self) -> list[BSplineBasisJAX]:
        return self.__basis 
    
    def control_points(self, params: Array | None = None) -> Array:
        if params is None:
            return self.__knots
        else:
            return self.__knots(params)

    def weights(self, params: Array | None = None) -> Array:
        if params is None:
            return self.__weights
        else:
            return self.__weights(params)
        
    
    def __init__(self, basis: list[BSplineBasisJAX], knots: Array | Callable, weights: Array | Callable, dparam: int, dembedding: int, bounds=None):
        

        if not ((dparam > 0 and callable(weights) and callable(knots)) or (dparam == 0 and not callable(knots) and not callable(weights))):
            raise Exception(
                "The weights and knots are callable iff the number of parameters is greater than 0.")

        d = len(basis)
        self.__basis = basis
        self.__knots = knots
        self.__weights = weights
 
        self.__N = [b.n for b in basis]

        if bounds is None:
            bounds = [(b.domain[0], b.domain[-1]) for b in basis]

            
        super(PatchNURBS, self).__init__(d, dembedding, dparam, bounds)
    
    def __call__(self, y: Array, params: Array | None = None, derivative: bool = False) -> Array:

        if not derivative:
            Bs = [b(y[:, i]).T for i, b in enumerate(self.__basis)]

            if self.dparam == 0:
                den = jnp.einsum('mi,i...->m...', Bs[0], self.__weights)
                for i in range(1, self.d):
                    den = jnp.einsum('mi,mi...->m...', Bs[i], den)
            else:
                def tmp(w, *bs):  # type: ignore
                    den = jnp.einsum('i,i...->...', bs[0], w)
                    for i in range(1, self.d):
                        den = jnp.einsum('i,i...->...', bs[i], den)

                    return den
                den = jax.vmap(
                    lambda x, *args: tmp(self.__weights(x), *args), (0))(params, *Bs)

            if self.dparam == 0:
                xs = jnp.einsum(
                    'mi,i...->m...', Bs[0], jnp.einsum('...i,...->...i', self.__knots, self.__weights))
                for i in range(1, self.d):
                    xs = jnp.einsum('mi,mi...->m...', Bs[i], xs)
            else:
                def tmp(w, k, *bs):
                    xs = jnp.einsum(
                        'i,i...->...', bs[0], jnp.einsum('...i,...->...i', k, w))
                    for i in range(1, self.d):
                        xs = jnp.einsum('i,i...->...', bs[i], xs)
                    return xs

                xs = jax.vmap(lambda x, *args: tmp(self.__weights(x),
                            self.__knots(x), *args))(params, *Bs)
            Gys = jnp.einsum('...i,...->...i', xs, 1/den)

            return Gys
        else:
            lst = []
            for d in range(self.d):
                lst.append(self._eval_derivative(y, params, d)[:, :, None])

            return jnp.concatenate(tuple(lst), -1)

    def __repr__(self) -> str:
        if self.d == 1:
            s = 'NURBS curve'
        elif self.d == 2:
            s = 'NURBS surface'
        elif self.d == 3:
            s = 'NURBS volume'
        else:
            s = 'NURBS instance'

        s += ' embedded in a ' + \
            str(self.dembedding)+'D space depending on ' + \
            str(self.dparam) + ' parameters.\n'
        s += 'Basis:\n'
        for b in self.__basis:
            s += str(b)+'\n'

        return s

    def __getitem__(self, key) -> Patch:

        if len(key) != self.d+self.dparam:
            raise Exception(
                'Invalid number of dimensions. It must equal the number of spacial dimensions + number of parameters.')

        axes = []
        basis_new = []
        bounds_new = []

        if key[-1] == Ellipsis:
            key = key[:-1]+[slice(None, None, None)]*self.dparam

        weights_mult = jnp.ones(self.__N)
        transform = False
        for k in range(self.d):
            id = key[k]
            if isinstance(id, int) or isinstance(id, float):
                if self.domain[k][0] <= id and id <= self.domain[k][1]:
                    axes.append(k)
                    B = self.__basis[k](np.array([id])).flatten()
                    s = tuple([None]*k+[slice(None, None, None)] +
                              [None]*(self.d-k-1))
                    B = B[s]
                    weights_mult = weights_mult*B
                    transform = True
                else:
                    raise Exception(
                        "Value must be inside the domain of the BSpline basis.")
            elif isinstance(id, slice):
                basis_new.append(self.__basis[k])
                start = id.start if id.start != None else self.domain[k][0]
                stop = id.stop if id.stop != None else self.domain[k][1]
                bounds_new.append((start, stop))
            else:
                raise Exception(
                    "Only slices, scalars and ellipsis are permitted")

        if self.dparam == 0:
            weights_new = jnp.sum(
                self.__weights*weights_mult, axis=tuple(axes))
            knots_new = jnp.sum(
                self.__knots*(self.__weights*weights_mult)[..., None], axis=tuple(axes))
            knots_new = knots_new/weights_new[..., None]
            param_new = 0
        else:
            weights_new = lambda *args: jnp.sum(
                self.__weights(*args)*weights_mult, axis=tuple(axes))
            knots_new = lambda *args: jnp.sum(self.__knots(*args)*(self.__weights(
                *args)*weights_mult)[..., None], axis=tuple(axes))/weights_new(*args)[..., None]

            params_take = []
            vect = []

            for k in range(self.dparam):
                id = key[k+self.d]
                if isinstance(id, int) or isinstance(id, float):
                    vect.append(float(id))
                elif isinstance(id, slice) and (id.start == None and id.stop == None and id.step == None):
                    params_take.append(k)
                    vect.append(0.0)
                else:
                    raise Exception(
                        "Only slices, scalars and ellipsis are permitted")

            E = np.eye(self.dparam)[tuple(params_take), :]
            vect = np.array(vect)

            if len(params_take) != 0:
                def weights_new(ps): return jnp.sum(self.__weights(jnp.einsum(
                    '...i,ij->...j', ps, E)+vect)*weights_mult, axis=tuple(axes))

                def knots_new(ps): return jnp.sum(self.__knots(jnp.einsum('...i,ij->...j', ps, E)+vect)*(self.__weights(jnp.einsum('...i,ij->...j',
                                                                                                                                   ps, E)+vect)*weights_mult)[..., None], axis=tuple(axes))/weights_new(jnp.einsum('...i,ij->...j', ps, E)+vect)[..., None]
            else:
                weights_new = jnp.sum(self.__weights(
                    vect)*weights_mult, axis=tuple(axes))
                knots_new = jnp.sum(self.__knots(
                    vect)*(self.__weights(vect)*weights_mult)[..., None], axis=tuple(axes))/weights_new[..., None]
            param_new = len(params_take)

        return PatchNURBS(basis_new, knots_new, weights_new, param_new, self.dembedding, bounds=bounds_new)

    def _eval_derivative(self, y: Array, params: Array | None, dim: int) -> Array:

        Bs = []
        dBs = []

        for i in range(len(self.__basis)):
            Bs.append(self.__basis[i](y[..., i]).T)
            dBs.append(self.__basis[i](y[:, i], derivative=(dim == i)).T)

        if self.dparam == 0:

            den = jnp.einsum('mi,i...->m...', Bs[0], self.__weights)
            for i in range(1, self.d):
                den = jnp.einsum('mi,mi...->m...', Bs[i], den)
            den = jnp.tile(den[..., None], self.dembedding)

            Dden = jnp.einsum('mi,i...->m...', dBs[0], self.__weights)
            for i in range(1, self.d):
                Dden = jnp.einsum('mi,mi...->m...', dBs[i], Dden)
            Dden = jnp.tile(Dden[..., None], self.dembedding)

            xs = jnp.einsum(
                'mi,i...->m...', Bs[0], jnp.einsum('...i,...->...i', self.__knots, self.__weights))
            for i in range(1, self.d):
                xs = jnp.einsum('mi,mi...->m...', Bs[i], xs)

            Dxs = jnp.einsum(
                'mi,i...->m...', dBs[0], jnp.einsum('...i,...->...i', self.__knots, self.__weights))
            for i in range(1, self.d):
                Dxs = jnp.einsum('mi,mi...->m...', dBs[i], Dxs)
        else:
            def tmp(w, *bs):
                den = jnp.einsum('i,i...->...', bs[0], w)
                for i in range(1, self.d):
                    den = jnp.einsum('i,i...->...', bs[i], den)
                den = jnp.tile(den[..., None], self.dembedding)
                return den

            tmp_jit = jax.vmap(lambda x, *args: tmp(self.__weights(x), *args))
            den = tmp_jit(params, *Bs)
            Dden = tmp_jit(params, *dBs)

            def tmp(w, k, *bs):
                xs = jnp.einsum(
                    'i,i...->...', bs[0], jnp.einsum('...i,...->...i', k, w))
                for i in range(1, self.__d):
                    xs = jnp.einsum('i,i...->...', bs[i], xs)
                return xs

            tmp_jit = jax.vmap(
                lambda x, *args: tmp(self.__weights(x), self.__knots(x), *args))
            xs = tmp_jit(params, *Bs)
            Dxs = tmp_jit(params, *dBs)

        return (Dxs*den-xs*Dden)/(den**2)

    def rotate(self, angles: Tuple[float]) -> PatchNURBS:
        """
        Rotate this object around the axes.

        Args:
            angles (Tuple[float]): the angles in radian.
        """
        if self.dparam != 0:
            pass
        else:
            if self.d==2:
                Rot = np.array([[np.cos(angles[0]), -np.sin(angles[0])],[np.sin(angles[0]), np.cos(angles[0])]])
            elif self.d==3:
                Rot = rotation_matrix_3d(angles)
            self.__knots = np.einsum('...n,mn->...m', self.__knots, Rot)

    def translate(self, offset: Tuple[float]) -> PatchNURBS:
        """
        translate the current object

        Args:
            offset (Tuple[float]): the offset.
        """
        
        if self.dparam != 0:
            pass
        else:
            self.__knots += np.array(offset, dtype=self.__knots.dtype)
            
    def affine_transformation(self, A: np.ndarray, b: np.ndarray) -> PatchNURBS:
        """_summary_

        Args:
            A (np.ndarray): _description_
            b (np.ndarray): _description_

        Returns:
            PatchNURBS: _description_
        """
        if self.dparam != 0:
            pass
        else:
            self.__knots = np.einsum('...n,mn->...m', self.__knots, A) + np.array(b, dtype=self.__knots.dtype)
  
    def copy(self) -> PatchNURBS:
        """
        Copy instance.

        Returns:
            PatchNURBS: the new instance.
        """
        
        return PatchNURBS(self.__basis, self.__knots.copy() if not callable(self.__knots) else self.__knots, self.weights.copy() if not callable(self.weights) else self.__weights, self.__dparam, self.__dembedding, self.__bounds)

class PatchTensorProduct(Patch):
    __basis: List[UnivariateFunctionBasis] # List of bases
    __control_pts: Array # control points mesh
    
    def __init__(self, basis: List[UnivariateFunctionBasis], control_points: Array | Callable, dparam: int, dembedding: int, bounds: List[Tuple[float, float]] | None = None):
        
        
        d = len(basis)
        if bounds is None:
            bounds = [b.domain for b in basis]
        self.__control_pts = control_points
        self.__basis = basis 
        
        super(PatchTensorProduct, self).__init__(d, dembedding, dparam, bounds)
        
    @property
    def basis(self) -> list[UnivariateFunctionBasis]:
        return self.__basis 
    
    def control_points(self, params: Array | None = None) -> Array:
        if params is None:
            return self.__control_pts
        else:
            return self.__control_pts(params)
        
    def __repr__(self) -> str:
        if self.d == 1:
            s = 'Tensor product curve'
        elif self.d == 2:
            s = 'Tensor product surface'
        elif self.d == 3:
            s = 'Tensor product volume'
        else:
            s = 'tensor product instance'

        s += ' embedded in a ' + \
            str(self.dembedding)+'D space depending on ' + \
            str(self.dparam) + ' parameters.\n'
        s += 'Basis:\n'
        for b in self.__basis:
            s += str(b)+'\n'

        return s

    def _eval_basis(self, y: Array, params: Array, derivatives: List[int]=[]) -> Array:
        """
        Evaluate the basis functions for a given input data.
    
        Args:
            y (Array): Input data array.
            params (Array): Parameters array.
            derivatives (List[int]): List of derivative orders to compute.
    
        Returns:
            Array: Array of evaluated basis functions.
        """
        Bs = [b(y[..., i], derivative=i in derivatives).T for i, b in enumerate(self.__basis)]

        if self.dparam == 0:
            xs = jnp.einsum('mi,i...->m...', Bs[0], self.__control_pts)
            for i in range(1, self.d):
                xs = jnp.einsum('mi,mi...->m...', Bs[i], xs)
        else:
            def tmp(k, *bs):
                xs = jnp.einsum(
                    'i,i...->...', bs[0], k)
                for i in range(1, self.d):
                    xs = jnp.einsum('i,i...->...', bs[i], xs)
                return xs

            xs = jax.vmap(lambda x, *args: tmp(self.__knots(x), *args))(params, *Bs)

        return xs
        
    def __call__(self, y: Array, params: Array | None = None, derivative: bool = False) -> Array:
        """
        Evaluate a geoemtry parametrization.

        Args:
            y (Array): Input data points in the reference domain.
            params (Array | None): Parameters for evaluating the geometry parametrization. If None, the default control points will be used.
            derivative (bool): Flag indicating whether to compute the jacobian of the parametrization.

        Returns:
            Array: Pointss in the computational domain or the Jacobian. Has shape `(..., d)` or `(..., d, d)`.

        Raises:
            None

        """

        if derivative: 
            lst = []
            for d in range(self.d):
                lst.append(self._eval_basis(y, params, [d])[:, :, None])

            return jnp.concatenate(tuple(lst), -1)
        else:
            return self._eval_basis(y, params)
            

    def __getitem__(self, key) -> Patch:
        raise NotImplementedError

    def rotate(self, angles: Tuple[float]):
        """
        Rotate this object around the axes.

        Args:
            angles (Tuple[float]): the angles in radian.
        """
        if self.dparam != 0:
            pass
        else:
            if self.d==2:
                Rot = np.array([[np.cos(angles[0]), -np.sin(angles[0])],[np.sin(angles[0]), np.cos(angles[0])]])
            elif self.d==3:
                Rot = rotation_matrix_3d(angles)
            self.__control_pts = np.einsum('...n,mn->...m', self.__control_pts, Rot)

    def translate(self, offset: Tuple[float]):
        """
        translate the current object

        Args:
            offset (Tuple[float]): the offset.
        """
        
        if self.dparam != 0:
            pass
        else:
            self.__control_pts += np.array(offset, dtype=self.__control_pts.dtype)
            
    def affine_transformation(self, A: np.ndarray, b: np.ndarray):
        """_summary_

        Args:
            A (np.ndarray): _description_
            b (np.ndarray): _description_

        Returns:
            PatchNURBS: _description_
        """
        if self.dparam != 0:
            pass
        else:
            self.__control_pts = np.einsum('...n,mn->...m', self.__control_pts, A) + np.array(b, dtype=self.__control_pts.dtype)
    
    def copy(self) -> Patch:
        """
        Create a copy of the current Patch object.

        Returns:
            Patch: A new Patch object with the same basis functions, control points,
                domain, and other attributes as the original Patch.
        """
        raise PatchTensorProduct(self.__basis, self.__control_pts, self.dparam, self.dembedding, self.domain)

    def distance_to_points(self, pts: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        
        if not all([isinstance(b, PiecewiseBernsteinBasisJAX) for b in self.__basis]):
            raise Exception("The patches must be Bezier")
        
        if self.d == 1:
            t_mins = []
            d_mins = []
            p_mins = []
            
            deg = self.__basis[0].deg 
            
            for i in range(self.__basis[0].knots.size-1):
                a = self.__basis[0].knots[i]
                b = self.__basis[0].knots[i+1]
                t_min, d_min, ps = distance_to_bezier_curve_simple(pts, self.__control_pts[(i*deg) : (i*deg+deg+1),:], deg, self.dembedding)
                t_min = (b-a)*t_min+a
                t_mins.append(t_min.reshape([-1,1]))
                d_mins.append(d_min.reshape([-1,1]))
                p_mins.append(ps.reshape([-1,self.dembedding,1]))
                
            t_mins = jnp.concatenate(tuple(t_mins), -1)
            d_mins = jnp.concatenate(tuple(d_mins), -1)
            p_mins = jnp.concatenate(tuple(p_mins), -1)
            
            idx = jnp.argmin(d_mins, axis=1)
            d_mins = d_mins[np.arange(d_mins.shape[0]),idx]
            t_mins = t_mins[np.arange(t_mins.shape[0]),idx]
            p_mins = p_mins[np.arange(p_mins.shape[0]),:,idx]

            return t_mins, d_mins, p_mins
        elif self.d == 2:
            
            u_mins = []
            v_mins = []
            d_mins = []
            p_mins = []
            
            deg = self.__basis[0].deg 
            
            for i in range(self.__basis[0].knots.size-1):
                for j in range(self.__basis[1].knots.size-1):
                    a0 = self.__basis[0].knots[i]
                    a1 = self.__basis[1].knots[j]
                    b0 = self.__basis[0].knots[i+1]
                    b1 = self.__basis[1].knots[j+1]
                    u_min, v_min, d_min, ps = distance_to_bezier_surface_simple(pts, self.__control_pts[(i*deg) : (i*deg+deg+1), (j*deg) : (j*deg+deg+1),:], deg, self.dembedding)
                    u_min = (b0-a0)*u_min+a0
                    v_min = (b1-a1)*v_min+a1
                    u_mins.append(u_min.reshape([-1,1]))
                    v_mins.append(v_min.reshape([-1,1]))
                    d_mins.append(d_min.reshape([-1,1]))
                    p_mins.append(ps.reshape([-1,self.dembedding,1]))
                
            u_mins = jnp.concatenate(tuple(u_mins), -1)
            v_mins = jnp.concatenate(tuple(v_mins), -1)
            d_mins = jnp.concatenate(tuple(d_mins), -1)
            p_mins = jnp.concatenate(tuple(p_mins), -1)
            
            idx = jnp.argmin(d_mins, axis=1)
            d_mins = d_mins[np.arange(d_mins.shape[0]),idx]
            u_mins = u_mins[np.arange(u_mins.shape[0]),idx]
            v_mins = v_mins[np.arange(v_mins.shape[0]),idx]
            p_mins = p_mins[np.arange(p_mins.shape[0]),:,idx]
            
            return u_mins, v_mins, d_mins, p_mins
                
class PatchBezier(PatchTensorProduct):
   
    def __init__(self, control_points: Array | Callable, shape: Tuple[int], degrees: Tuple[int], dparam: int, dembedding: int, bounds: List[Tuple[float, float]]):
        
        d = len(shape)
        assert d == len(degrees), "Invalid arguments: shape and degrees length does not match."
        
        bases = [ k for k in range(d)]
        
        super(PatchBezier, self).__init__(bases, control_points, dparam, dembedding, bounds)
    def __repr__(self) -> str:
        if self.d == 1:
            s = 'Bezier curve'
        elif self.d == 2:
            s = 'Bezier surface'
        elif self.d == 3:
            s = 'Bezier volume'
        else:
            s = 'Bezier instance'

        s += ' embedded in a ' + \
            str(self.dembedding)+'D space depending on ' + \
            str(self.dparam) + ' parameters.\n'
        s += 'Basis:\n'
        for b in self.__basis:
            s += str(b)+'\n'

        return s
    
    def copy(self) -> Patch:
        """
        Create a copy of the current Patch object.

        Returns:
            Patch: A new Patch object with the same basis functions, control points,
                domain, and other attributes as the original Patch.
        """
        raise PatchTensorProduct(self.__basis, self.__control_pts, self.dparam, self.dembedding, self.domain)
        
    #def __init__(self, bases: List[])

# class PatchNURBSParam(Patch):
#     __d: int
#     __dembedding: int
#     __dparam: int
#     __basis: list[BSplineBasisJAX]
#     __weights: Callable | Array
#     __knots: Callable | Array
#     __bounds: list[tuple[float, float]]
#     __N: list[int]
# 
#     @property
#     def d(self) -> int:
#         """
#         The dimension of the manifold.
# 
#         Returns:
#             int: the number of dimensions.
#         """
#         return self.__d
# 
#     @property 
#     def dembedding(self) -> int:
#         """
#         Returns the number of dimensions where the patch lives.
# 
#         Returns:
#             int: the number of dimensions.
#         """
#         return self.__dembedding
#     
#     @property
#     def domain(self) -> list[tuple[float, float]]:
#         """
#         The definition domain of the patch. Without parameters.
# 
#         Returns:
#             list[tuple[float, float]]: tuple of intervals.
#         """
#         return self.__bounds
# 
#     @property
#     def basis(self) -> list[BSplineBasisJAX]:
#         return self.__basis 
#     
#     def knots(self, params: Array | None = None) -> Array:
#         if params is None:
#             return self.__knots
#         else:
#             return self.__knots(params)
# 
#     def weights(self, params: Array | None = None) -> Array:
#         if params is None:
#             return self.__weights
#         else:
#             return self.__weights(params)
# 
#     def __init__(self, basis: list[BSplineBasisJAX], knots: Array | Callable, weights: Array | Callable, dparam: int, dembedding: int, rand_key: jax.random.PRNGKey, bounds=None):
#         super(PatchNURBSParam, self).__init__(rand_key)
# 
#         if not ((dparam > 0 and callable(weights) and callable(knots)) or (dparam == 0 and not callable(knots) and not callable(weights))):
#             raise Exception(
#                 "The weights and knots are callable iff the number of parameters is greater than 0.")
# 
#         self.__d = len(basis)
#         self.__basis = basis
#         self.__knots = knots
#         self.__weights = weights
#         self.__dembedding = dembedding
#         self.__dparam = dparam
#         self.__N = [b.n for b in basis]
# 
#         if bounds == None:
#             self.__bounds = [(b.knots[0], b.knots[-1]) for b in basis]
#         else:
#             self.__bounds = bounds
# 
#     def __call__(self, y: Array, params: Array | None = None, differential: bool = False) -> Array | tuple[Array, Array]:
# 
#         Bs = [b(y[:, i]).T for i, b in enumerate(self.__basis)]
# 
#         if self.__dparam == 0:
#             den = jnp.einsum('mi,i...->m...', Bs[0], self.__weights)
#             for i in range(1, self.__d):
#                 den = jnp.einsum('mi,mi...->m...', Bs[i], den)
#         else:
#             def tmp(w, *bs):  # type: ignore
#                 den = jnp.einsum('i,i...->...', bs[0], w)
#                 for i in range(1, self.__d):
#                     den = jnp.einsum('i,i...->...', bs[i], den)
# 
#                 return den
#             den = jax.vmap(
#                 lambda x, *args: tmp(self.__weights(x), *args), (0))(params, *Bs)
# 
#         if self.__dparam == 0:
#             xs = jnp.einsum(
#                 'mi,i...->m...', Bs[0], jnp.einsum('...i,...->...i', self.__knots, self.__weights))
#             for i in range(1, self.__d):
#                 xs = jnp.einsum('mi,mi...->m...', Bs[i], xs)
#         else:
#             def tmp(w, k, *bs):
#                 xs = jnp.einsum(
#                     'i,i...->...', bs[0], jnp.einsum('...i,...->...i', k, w))
#                 for i in range(1, self.__d):
#                     xs = jnp.einsum('i,i...->...', bs[i], xs)
#                 return xs
# 
#             xs = jax.vmap(lambda x, *args: tmp(self.__weights(x),
#                           self.__knots(x), *args))(params, *Bs)
#         Gys = jnp.einsum('...i,...->...i', xs, 1/den)
# 
#         if differential:
#             DGys = self._eval_omega(y, params)
#             Weights = 1
#             if self.__d == 3 and self.__dembedding == 3:
#                 diff = jnp.abs(DGys[:, 0, 0]*DGys[:, 1, 1]*DGys[:, 2, 2] + DGys[:, 0, 1]*DGys[:, 1, 2]*DGys[:, 2, 0]+DGys[:, 0, 2]*DGys[:, 1, 0]*DGys[:, 2, 1] -
#                                DGys[:, 0, 2]*DGys[:, 1, 1]*DGys[:, 2, 0] - DGys[:, 0, 0]*DGys[:, 1, 2]*DGys[:, 2, 1] - DGys[:, 0, 1]*DGys[:, 1, 0]*DGys[:, 2, 2])*Weights
#             elif self.__d == 2 and self.__dembedding == 2:
#                 diff = jnp.abs(DGys[:, 0, 0]*DGys[:, 1, 1] -
#                                DGys[:, 0, 1]*DGys[:, 1, 0])*Weights
#             elif self.__d == 2 and self.__dembedding == 3:
#                 def tmp(tangents):
#                     result1 = tangents[..., 1, 0]*tangents[..., 2,
#                                                            1]-tangents[..., 2, 0]*tangents[..., 1, 1]
#                     result2 = tangents[..., 2, 0]*tangents[..., 0,
#                                                            1]-tangents[..., 0, 0]*tangents[..., 2, 1]
#                     result3 = tangents[..., 0, 0]*tangents[..., 1,
#                                                            1]-tangents[..., 1, 0]*tangents[..., 0, 1]
#                     return jnp.concatenate((result1[..., None], result2[..., None], result3[..., None]), -1)
#                 diff = tmp(DGys)  # jnp.einsum('ij,i->ij',tmp(DGys),Weights)
#             elif self.__d == 1:
#                 diff = DGys[..., :, 0]*Weights
#             else:
#                 diff = DGys  # jnp.einsum('ij,i->ij',DGys,Weights)
# 
#         if differential:
#             return Gys, diff
#         else:
#             return Gys
# 
#     def __repr__(self) -> str:
#         if self.__d == 1:
#             s = 'NURBS curve'
#         elif self.__d == 2:
#             s = 'NURBS surface'
#         elif self.__d == 3:
#             s = 'NURBS volume'
#         else:
#             s = 'NURBS instance'
# 
#         s += ' embedded in a ' + \
#             str(self.__dembedding)+'D space depending on ' + \
#             str(self.__dparam) + ' parameters.\n'
#         s += 'Basis:\n'
#         for b in self.__basis:
#             s += str(b)+'\n'
# 
#         return s
# 
#     def __getitem__(self, key) -> Patch:
# 
#         if len(key) != self.__d+self.__dparam:
#             raise Exception(
#                 'Invalid number of dimensions. It must equal the number of spacial dimensions + number of parameters.')
# 
#         axes = []
#         basis_new = []
#         bounds_new = []
# 
#         if key[-1] == Ellipsis:
#             key = key[:-1]+[slice(None, None, None)]*self.__dparam
# 
#         weights_mult = jnp.ones(self.__N)
#         transform = False
#         for k in range(self.__d):
#             id = key[k]
#             if isinstance(id, int) or isinstance(id, float):
#                 if self.__bounds[k][0] <= id and id <= self.__bounds[k][1]:
#                     axes.append(k)
#                     B = self.__basis[k](np.array([id])).flatten()
#                     s = tuple([None]*k+[slice(None, None, None)] +
#                               [None]*(self.__d-k-1))
#                     B = B[s]
#                     weights_mult = weights_mult*B
#                     transform = True
#                 else:
#                     raise Exception(
#                         "Value must be inside the domain of the BSpline basis.")
#             elif isinstance(id, slice):
#                 basis_new.append(self.__basis[k])
#                 start = id.start if id.start != None else self.__bounds[k][0]
#                 stop = id.stop if id.stop != None else self.__bounds[k][1]
#                 bounds_new.append((start, stop))
#             else:
#                 raise Exception(
#                     "Only slices, scalars and ellipsis are permitted")
# 
#         if self.__dparam == 0:
#             weights_new = jnp.sum(
#                 self.__weights*weights_mult, axis=tuple(axes))
#             knots_new = jnp.sum(
#                 self.__knots*(self.__weights*weights_mult)[..., None], axis=tuple(axes))
#             knots_new = knots_new/weights_new[..., None]
#             param_new = 0
#         else:
#             weights_new = lambda *args: jnp.sum(
#                 self.__weights(*args)*weights_mult, axis=tuple(axes))
#             knots_new = lambda *args: jnp.sum(self.__knots(*args)*(self.__weights(
#                 *args)*weights_mult)[..., None], axis=tuple(axes))/weights_new(*args)[..., None]
# 
#             params_take = []
#             vect = []
# 
#             for k in range(self.__dparam):
#                 id = key[k+self.__d]
#                 if isinstance(id, int) or isinstance(id, float):
#                     vect.append(float(id))
#                 elif isinstance(id, slice) and (id.start == None and id.stop == None and id.step == None):
#                     params_take.append(k)
#                     vect.append(0.0)
#                 else:
#                     raise Exception(
#                         "Only slices, scalars and ellipsis are permitted")
# 
#             E = np.eye(self.__dparam)[tuple(params_take), :]
#             vect = np.array(vect)
# 
#             if len(params_take) != 0:
#                 def weights_new(ps): return jnp.sum(self.__weights(jnp.einsum(
#                     '...i,ij->...j', ps, E)+vect)*weights_mult, axis=tuple(axes))
# 
#                 def knots_new(ps): return jnp.sum(self.__knots(jnp.einsum('...i,ij->...j', ps, E)+vect)*(self.__weights(jnp.einsum('...i,ij->...j',
#                                                                                                                                    ps, E)+vect)*weights_mult)[..., None], axis=tuple(axes))/weights_new(jnp.einsum('...i,ij->...j', ps, E)+vect)[..., None]
#             else:
#                 weights_new = jnp.sum(self.__weights(
#                     vect)*weights_mult, axis=tuple(axes))
#                 knots_new = jnp.sum(self.__knots(
#                     vect)*(self.__weights(vect)*weights_mult)[..., None], axis=tuple(axes))/weights_new[..., None]
#             param_new = len(params_take)
# 
#         return PatchNURBSParam(basis_new, knots_new, weights_new, param_new, self.__dembedding, self.rand_key, bounds=bounds_new)
# 
#     def _eval_derivative(self, y: Array, params: Array | None, dim: int) -> Array:
# 
#         Bs = []
#         dBs = []
# 
#         for i in range(len(self.__basis)):
#             Bs.append(self.__basis[i](y[..., i]).T)
#             dBs.append(self.__basis[i](y[:, i], derivative=(dim == i)).T)
# 
#         if self.__dparam == 0:
# 
#             den = jnp.einsum('mi,i...->m...', Bs[0], self.__weights)
#             for i in range(1, self.__d):
#                 den = jnp.einsum('mi,mi...->m...', Bs[i], den)
#             den = jnp.tile(den[..., None], self.__dembedding)
# 
#             Dden = jnp.einsum('mi,i...->m...', dBs[0], self.__weights)
#             for i in range(1, self.__d):
#                 Dden = jnp.einsum('mi,mi...->m...', dBs[i], Dden)
#             Dden = jnp.tile(Dden[..., None], self.__dembedding)
# 
#             xs = jnp.einsum(
#                 'mi,i...->m...', Bs[0], jnp.einsum('...i,...->...i', self.__knots, self.__weights))
#             for i in range(1, self.__d):
#                 xs = jnp.einsum('mi,mi...->m...', Bs[i], xs)
# 
#             Dxs = jnp.einsum(
#                 'mi,i...->m...', dBs[0], jnp.einsum('...i,...->...i', self.__knots, self.__weights))
#             for i in range(1, self.__d):
#                 Dxs = jnp.einsum('mi,mi...->m...', dBs[i], Dxs)
#         else:
#             def tmp(w, *bs):
#                 den = jnp.einsum('i,i...->...', bs[0], w)
#                 for i in range(1, self.__d):
#                     den = jnp.einsum('i,i...->...', bs[i], den)
#                 den = jnp.tile(den[..., None], self.__dembedding)
#                 return den
# 
#             tmp_jit = jax.vmap(lambda x, *args: tmp(self.__weights(x), *args))
#             den = tmp_jit(params, *Bs)
#             Dden = tmp_jit(params, *dBs)
# 
#             def tmp(w, k, *bs):
#                 xs = jnp.einsum(
#                     'i,i...->...', bs[0], jnp.einsum('...i,...->...i', k, w))
#                 for i in range(1, self.__d):
#                     xs = jnp.einsum('i,i...->...', bs[i], xs)
#                 return xs
# 
#             tmp_jit = jax.vmap(
#                 lambda x, *args: tmp(self.__weights(x), self.__knots(x), *args))
#             xs = tmp_jit(params, *Bs)
#             Dxs = tmp_jit(params, *dBs)
# 
#         return (Dxs*den-xs*Dden)/(den**2)
# 
#     def GetJacobian(self, y: Array, params: Array | None = None) -> jax.Array:
#         """
#         Evaluate the Jacobian of the geometry transformation
# 
#         Args:
#             y (jax.numpy.array): the positions in the reference domain. Has the shape N x d.
# 
#         Returns:
#             jax.numpy.array: _description_
#         """
# 
#         lst = []
#         for d in range(self.__d):
#             lst.append(self._eval_derivative(y, params, d)[:, :, None])
# 
#         return jnp.concatenate(tuple(lst), -1)
# 
#     def GetMetricTensors(self, y: Array, params: Array | None = None) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
# 
#         DGys = self.GetJacobian(y, params)
#         Inv = jnp.linalg.inv(DGys)
#         omega = jnp.abs(jnp.linalg.det(DGys))
#         K = jnp.einsum('mij,mjk,m->mik', Inv,
#                        jnp.transpose(Inv, [0, 2, 1]), omega)
#         Gr = Inv
# 
#         return (omega, DGys, Gr, K)
# 
#     def _eval_omega(self, y: Array, params: Array | None = None):
#         """
#         Evaluate the derivative of the parametrization (jacobian).
#         y[...,i,j] = \partial_y
# 
#         Args:
#             y (numpy.array): the points
# 
#         Returns:
#             numpy.array: _description_
#         """
#         lst = []
#         for d in range(self.__d):
#             lst.append(self._eval_derivative(y, params, d)[:, :, None])
# 
#         return jnp.concatenate(tuple(lst), -1)
# 
#     def importance_sampling(self, N, pdf=None):
# 
#         if pdf is None:
#             def pdf(x): return 1.0
# 
#         vol_ref = 1.0
#         for i in self.__bounds:
#             vol_ref *= i[1]-i[0]
# 
#         ys = np.random.rand(N, self.__d)*np.array(
#             [i[1]-i[0] for i in self.__bounds]) + np.array([i[0] for i in self.__bounds])
#         Gys = self.__call__(ys)
# 
#         DGys = self._eval_omega(ys)
# 
#         if self.__d == 3 and self.__dembedding == 3:
#             diff = np.abs(DGys[:, 0, 0]*DGys[:, 1, 1]*DGys[:, 2, 2] + DGys[:, 0, 1]*DGys[:, 1, 2]*DGys[:, 2, 0]+DGys[:, 0, 2]*DGys[:, 1, 0]*DGys[:, 2, 1] -
#                           DGys[:, 0, 2]*DGys[:, 1, 1]*DGys[:, 2, 0] - DGys[:, 0, 0]*DGys[:, 1, 2]*DGys[:, 2, 1] - DGys[:, 0, 1]*DGys[:, 1, 0]*DGys[:, 2, 2])
#         elif self.__d == 2 and self.__dembedding == 2:
#             diff = np.abs(DGys[:, 0, 0]*DGys[:, 1, 1] -
#                           DGys[:, 0, 1]*DGys[:, 1, 0])
#         elif self.__d == 2 and self.__dembedding == 3:
#             diff = tangent2normal_3d(DGys)
#         elif self.__d == 1:
#             diff = DGys[..., :, 0]
#         else:
#             diff = DGys
#         return Gys, diff*vol_ref/N
# 
#     def quadrature(self, N=32, knots='leg'):
# 
#         Knots = [(np.polynomial.legendre.leggauss(N)[0]+1)*0.5*(self.bounds[i]
#                                                                 [1]-self.bounds[i][0])+self.bounds[i][0] for i in range(self.d)]
#         Ws = [np.polynomial.legendre.leggauss(
#             N)[1]*0.5*(self.bounds[i][1]-self.bounds[i][0]) for i in range(self.d)]
#         Knots = np.meshgrid(*Knots)
#         ys = np.concatenate(tuple([k.flatten()[:, None] for k in Knots]), -1)
# 
#         Weights = Ws[0]
#         for i in range(1, self.d):
#             Weights = np.kron(Weights, Ws[i])
# 
#         Gys = self.__call__(ys)
# 
#         DGys = self._eval_omega(ys)
# 
#         if self.d == 3 and self.dembedding == 3:
#             diff = np.abs(DGys[:, 0, 0]*DGys[:, 1, 1]*DGys[:, 2, 2] + DGys[:, 0, 1]*DGys[:, 1, 2]*DGys[:, 2, 0]+DGys[:, 0, 2]*DGys[:, 1, 0]*DGys[:, 2, 1] -
#                           DGys[:, 0, 2]*DGys[:, 1, 1]*DGys[:, 2, 0] - DGys[:, 0, 0]*DGys[:, 1, 2]*DGys[:, 2, 1] - DGys[:, 0, 1]*DGys[:, 1, 0]*DGys[:, 2, 2])*Weights
#         elif self.d == 2 and self.dembedding == 2:
#             diff = np.abs(DGys[:, 0, 0]*DGys[:, 1, 1] -
#                           DGys[:, 0, 1]*DGys[:, 1, 0])*Weights
#         elif self.d == 2 and self.dembedding == 3:
#             diff = np.einsum('ij,i->ij', tangent2normal_3d(DGys), Weights)
#         elif self.d == 1:
#             diff = DGys[..., :, 0]*Weights
#         else:
#             diff = np.einsum('ij,i->ij', DGys, Weights)
#         return Gys, diff
# 
#     def importance_sampling_2d(self, N, pdf=None):
# 
#         if pdf is None:
#             def pdf(x): return 1.0
# 
#         ys = np.random.rand(N, self.d)
# 
#         Gys = self.__call__(ys)
# 
#         DGys = self._eval_omega(ys)
# 
#         det = np.abs(DGys[:, 0, 0]*DGys[:, 1, 1] - DGys[:, 0, 1]*DGys[:, 1, 0])
# 
#         return Gys, det
# 
#     def importance_sampling_3d(self, N, pdf=None, bounds=None):
# 
#         if pdf is None:
#             def pdf(x): return 1.0
#         if bounds == None:
#             bounds = ((0, 1), (0, 1), (0, 1))
# 
#         ys = np.random.rand(N, self.d)*np.array([bounds[0][1]-bounds[0][0], bounds[1][1]-bounds[1]
#                                                  [0], bounds[2][1]-bounds[2][0]]) + np.array([bounds[0][0], bounds[1][0], bounds[2][0]])
#         Gys = self.__call__(ys)
# 
#         DGys = self._eval_omega(ys)
# 
#         det = np.abs(DGys[:, 0, 0]*DGys[:, 1, 1]*DGys[:, 2, 2] + DGys[:, 0, 1]*DGys[:, 1, 2]*DGys[:, 2, 0]+DGys[:, 0, 2]*DGys[:, 1, 0]*DGys[:, 2, 1] -
#                      DGys[:, 0, 2]*DGys[:, 1, 1]*DGys[:, 2, 0] - DGys[:, 0, 0]*DGys[:, 1, 2]*DGys[:, 2, 1] - DGys[:, 0, 1]*DGys[:, 1, 0]*DGys[:, 2, 2])
#         det = det/det.size*(bounds[0][1]-bounds[0][0]) * \
#             (bounds[1][1]-bounds[1][0])*(bounds[2][1]-bounds[2][0])
# 
#         return Gys, det
# 
#     def sample_inside(self, N, pdf=None):
# 
#         if pdf is None:
#             ys = np.random.rand(N, self.d)
#             # ys = jax.random.uniform(self.rand_key, (N, self.d))
#             xs = self.__call__(ys)
#         else:
#             pass
#             # Gy = self.__call__(ys)
#         return xs
# 
#     def rotate(self, angles: Tuple[float]):
#         """
#         Rotate this object around the axes.
# 
#         Args:
#             angles (Tuple[float]): the angles in radian.
#         """
#         if self.__dparam != 0:
#             pass
#         else:
#             if self.__d==2:
#                 Rot = np.array([[np.cos(angles[0]), -np.sin(angles[0])],[np.sin(angles[0]), np.cos(angles[0])]])
#             elif self.__d==3:
#                 Rot = rotation_matrix_3d(angles)
#             self.__knots = np.einsum('...n,mn->...m', self.__knots, Rot)
# 
#     def translate(self, offset: Tuple[float]):
#         """
#         translate the current object
# 
#         Args:
#             offset (Tuple[float]): the offset.
#         """
#         
#         if self.__dparam != 0:
#             pass
#         else:
#             self.__knots += np.array(offset, dtype=self.__knots.dtype)
#             
#         
#                 
#     def sample_boundary(self, d, end, N, normalize=True, pdf=None):
# 
#         if pdf == None:
#             y = np.random.rand(N, self.d)
#             y[:, d] = end
#             Bs = [b(y[:, i]) for i, b in enumerate(self.basis)]
#             dBs = [b(y[:, i], derivative=True)
#                    for i, b in enumerate(self.basis)]
# 
#             pts = self.__call__(y)
#             pts_tangent = []
#             for i in range(self.d):
#                 if i != d:
#                     ds = [False]*self.d
#                     ds[i] = True
# 
#                     v = self._eval_derivative(y, i)
#                     if normalize:
#                         v = v/np.tile(np.linalg.norm(v, axis=1,
#                                       keepdims=True), self.d)
#                     pts_tangent += [v[:, None, :]]
#             # v = self._eval_derivative(y, d)
#             # norm = v/np.tile(np.linalg.norm(v,axis = 1, keepdims=True),self.d)
#             return pts, np.concatenate(tuple(pts_tangent), 1)  # , norm
#         
# 
# 