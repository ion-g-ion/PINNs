# import jax
# import jax.numpy as jnp
# import numpy as np
# 
# class BSplineBasis():
#     
#     def __init__(self, knots, deg):
#         self.deg = deg
#         self.n = knots.size+deg-1
#         self.knots = np.concatenate(( np.ones(deg)*knots[0], knots , np.ones(deg)*knots[-1] ))
#         
#     def _eval_basis(self,x):
# 
#         result = np.zeros((x.shape[0],self.knots.size-1))
#         
#         for i in range(result.shape[1]):
#             if self.knots[-1]<=self.knots[i+1] and self.knots[i]<self.knots[-1]:
#                 idx = np.logical_and(x>=self.knots[i], x<=self.knots[i+1]) 
#             else:
#                 idx = np.logical_and(x>=self.knots[i], x<self.knots[i+1])
#             # result = result.at[idx,i].set(1.0)
#             result[idx,i] = 1.0
#         
#         for d in range(self.deg):
#             for i in range(result.shape[1]-d-1):
#                 a = result[:,i]*(x-self.knots[i])/(self.knots[i+d+1]-self.knots[i])
#                 b = result[:,i+1]*(self.knots[i+d+2]-x)/(self.knots[i+d+2]-self.knots[i+1])
#                 result_new = result.copy()
#                 # result_new = result_new.at[:,i].set(0.0)
#                 result_new[:,i] = 0.0
#                 if (self.knots[i+d+1]-self.knots[i]) != 0:
#                     # result_new = result_new.at[:,i].set( result_new[:,i] + a )
#                     result_new[:,i] = result_new[:,i] + a
#                 if (self.knots[i+d+2]-self.knots[i+1])!=0:
#                     # result_new = result_new.at[:,i].set( result_new[:,i] + b )
#                     result_new[:,i] = result_new[:,i] + b
#                 result = result_new.copy() 
#         # result[x==self.knots[-1],self.n-1] = 1.0
#         return result[:,:self.n]
#            
#     def _eval_basis_derivative(self,x):
# 
#         result = np.zeros((x.shape[0],self.knots.size-1))
#         
#         for i in range(result.shape[1]):
#             idx = np.logical_and(x>=self.knots[i], x<=self.knots[i+1]) if self.knots[-1]<=self.knots[i+1] and self.knots[i]<self.knots[-1] else jnp.logical_and(x>=self.knots[i], x<self.knots[i+1])
#             # result = result.at[idx,i].set(1.0)
#             result[idx,i] = 1.0
#         #result[x==self.knots[-1],-1] = 1.0
#         
#         for d in range(self.deg):
#             for i in range(result.shape[1]-d-1):
#                 if d == self.deg-1:
#                     a = self.deg*result[:,i]*(1)/(self.knots[i+d+1]-self.knots[i])
#                     b = self.deg*result[:,i+1]*(-1)/(self.knots[i+d+2]-self.knots[i+1])
#                 else:
#                     a = result[:,i]*(x-self.knots[i])/(self.knots[i+d+1]-self.knots[i])
#                     b = result[:,i+1]*(self.knots[i+d+2]-x)/(self.knots[i+d+2]-self.knots[i+1])
#                 result_new = result.copy()
#                 # result_new = result_new.at[:,i].set(0.0)
#                 result_new[:,i] = 0.0
#                 if (self.knots[i+d+1]-self.knots[i]) != 0:
#                     # result_new  = result_new.at[:,i].set(result_new[:,i] + a)
#                     result_new[:,i] = result_new[:,i] + a
#                 if (self.knots[i+d+2]-self.knots[i+1])!=0:
#                     # result_new = result_new.at[:,i].set(result_new[:,i] + b)
#                     result_new[:,i] = result_new[:,i] + b
#                 result = result_new.copy() 
#                 
#         # result[x==self.knots[-1],self.n-1] = 1.0
#         return result[:,:self.n]
#             
#     def __call__(self, x, derivative = False):
#         """
#         Evaluates the basis at the given points x.
#         Args:
#             x (torch.tensor): vector of size M
#             derivative (bool, optional): evaluate the derivative of the basis or not. Defaults to False.
#         Returns:
#             torch.tensor: a matrix of size MxN
#         """
#         if derivative:
#             return self._eval_basis_derivative(x)
#         else:
#             return self._eval_basis(x)
#    
#     def interpolation_points(self):
# 
#         pts = np.array([np.sum(self.knots[i+1:i+self.deg+1]) for i in range(self.n)])/(self.deg)
#         Mat = self.__call__(pts)
#         return pts, Mat


from scipy.interpolate import BSpline
import scipy.interpolate
import scipy.special 
import numpy as np
import abc

class BSplineBasis:
    
    def __init__(self,knots,deg,ends = (True,True)):
        """
        Univariate BSpline basis class.

        Args:
            knots (numpy.array): _description_
            deg (int): _description_
            ends (tuple, optional): _description_. Defaults to (True,True).
        """
        self.N=knots.size+deg-1
        self.deg=deg
        self.knots=np.hstack( ( np.ones(deg)*knots[0] , knots , np.ones(deg)*knots[-1] ) )
        self.spl = []
        self.dspl = []
        self.interval = (np.min(knots),np.max(knots))
        for i in range(self.N):
            c=np.zeros(self.N)
            c[i]=1
            self.spl.append(BSpline(self.knots,c,self.deg))
            self.dspl.append(scipy.interpolate.splder( BSpline(self.knots,c,self.deg) ))
        
        self.compact_support_bsp = np.zeros((self.N,2))
        for i in range(self.N):
            self.compact_support_bsp[i,0] = self.knots[i]
            self.compact_support_bsp[i,1] = self.knots[i+self.deg+1]
            
        int_bsp_bsp = np.zeros((self.N,self.N))
        int_bsp = np.zeros((self.N,1))
        # int_bsp_v = np.zeros((self.Nz,1))
        
        Pts, Ws =np.polynomial.legendre.leggauss(20)
        for i in range(self.N):
            a=self.compact_support_bsp[i,0]
            b=self.compact_support_bsp[i,1]

            for k in range(self.knots.size-1):
                if self.knots[k]>=a and self.knots[k+1]<=b:
                    pts = self.knots[k]+(Pts+1)*0.5*(self.knots[k+1]-self.knots[k])
                    ws = Ws*(self.knots[k+1]-self.knots[k])/2
                    int_bsp[i,0] += np.sum( self.__call__(pts,i) * ws )
                    
            for j in range(i,self.N):
                a=self.compact_support_bsp[j,0]
                b=self.compact_support_bsp[i,1]
                if b>a:
                    for k in range(self.knots.size-1):
                        if self.knots[k]>=a and self.knots[k+1]<=b:
                            pts = self.knots[k]+(Pts+1)*0.5*(self.knots[k+1]-self.knots[k])
                            ws = Ws*(self.knots[k+1]-self.knots[k])/2
                            int_bsp_bsp[i,j] += np.sum(  self.__call__(pts,i) *self.__call__(pts,j) * ws )
                            # int_bspp[i,j] += np.sum( self.bspp(pts)[i,:]* self.bspp(pts)[j,:]*ws )
                    if i!=j:
                        int_bsp_bsp[j,i] = int_bsp_bsp[i,j]
                        # int_bspp[j,i] = int_bspp[i,j]
                    
        
        self.int_bsp_bsp = int_bsp_bsp
        # self.int_bspp_bspp = int_bspp
        self.int_bsp = int_bsp
        
    
    def __call__(self,x,i=None,derivative=False):
        if i==None:
            if derivative:
                ret = np.array([self.dspl[i](x) for i in range(self.N)])
                return ret
            else:
                ret = np.array([self.spl[i](x) for i in range(self.N)])
                return ret
        else:
            if derivative:
                 return self.dspl[i](x)
            else:
                return self.spl[i](x)
            
    def __repr__(self) -> str:
        return 'B-Spline basis of degree '+str(self.deg)+' and dimension '+str(self.N)
     
    def greville(self):
        return np.array([np.sum(self.knots[i+1:i+self.deg+1]) for i in range(self.N)])/(self.deg)
        # return np.array([np.sum(self.knots[i+2:i+self.deg+2]) for i in range(self.N)])/(self.deg-1)
        
    def interpolating_points(self):
        pts = np.array([np.sum(self.knots[i+1:i+self.deg+1]) for i in range(self.N)])/(self.deg)
        Mat = self.__call__(pts)
        return pts, Mat
        
    def abscissae(self):
        return self.greville()

    def quadrature_points(self,mult = 1):
        pts = []
        ws = []
        Pts, Ws = np.polynomial.legendre.leggauss(mult)
        for k in range(self.knots.size-1):
            if self.knots[k+1]>self.knots[k]:
                pts += list(self.knots[k]+(Pts+1)*0.5*(self.knots[k+1]-self.knots[k]))
                ws += list(Ws*(self.knots[k+1]-self.knots[k])/2)
        pts = np.array(pts)
        ws = np.array(ws)
        
        return pts, ws
            
    def eval_all(self,c,x):
        c=np.hstack((c,np.zeros(self.deg-2)))
        return BSpline(self.knots,c,self.deg)(x)
    
    def plot(self,derivative = False):
        
        for i in range(self.N):
            x=np.linspace(self.compact_support_bsp[i,0],self.compact_support_bsp[i,1],500)
            # plt.plot(x,self.__call__(x,i,derivative))
    def derivative(self):
        bd = scipy.interpolate.splder(BSpline(self.knots,np.zeros(self.N+self.deg-1)+1,self.deg))
        return BSplineBasis(np.unique(bd.t), bd.k)
    def integrate(self):
        BI = np.zeros((self.N+1,self.N+1))
        BII = np.zeros((self.N+1,self.N+1))
        a = self.knots[0]
        b = self.knots[-1]
        pts, ws =np.polynomial.legendre.leggauss(128)
        pts = a+(pts+1)*0.5*(b-a)
        for i in range(self.N+1):
            for j in range(i,self.N+1):
                BI[i,j] = np.sum( self.eval_single(i,pts)*self.eval_single(j,pts)*ws*(b-a)/2 )
                BI[j,i] = BI[i,j]
        return BII,BI


import jax.numpy as jnp
import jax 
import jax.lax

class JaxPiecewiseLinear():
    
    def __init__(self, knots):
        self.N = knots.size
        self.knots = knots
        
    def __call__(self, x, dofs):
        
        ret = 0*x
        for i in range(self.N-1):
            idx = jnp.logical_and(x>=self.konts[i],x<self.knots[i+1])
            ret = ret.at[idx].set( 1 )

class UnivariateFunctionBasis(metaclass=abc.ABCMeta):
    __interval: tuple[float, float]
    
    def __init__(self, interval: tuple[float, float]):
        self.__interval = interval

    @property 
    def domain(self) -> tuple[float, float]:
        return self.__interval
    
    @classmethod
    def __subclasshook__(cls, subclass):
            return (hasattr(subclass, '__repr__') and 
                callable(subclass.__repr__) and 
                hasattr(subclass, '__call__') and 
                callable(subclass.__call__) and 
                hasattr(subclass, 'interpolating_points') and 
                callable(subclass.interpolating_points) and 
                hasattr(subclass, 'quadrature_points') and 
                callable(subclass.quadrature_points))  or NotImplemented

    @abc.abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError
    
    @abc.abstractmethod
    def __call__(self, x: jax.Array, derivative: bool=False) -> jax.Array:
        raise NotImplementedError
    
    @abc.abstractmethod
    def interpolating_points(self) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def quadrature_points(self, deg: int) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError
    
    
    

class PiecewiseBernsteinBasisJAX(UnivariateFunctionBasis):
    
    __deg: int
    __n: int 
    __knots: jax.Array 
    __coeffs: np.array 
    
    def __init__(self, knots: np.ndarray, deg: int):
        """
        Create a  piecewise Bernstein polynomial basis.

        Args:
            knots (np.ndarray): the knots of the basis.
            deg (int): the degree of the basis.
        """
        a, b = float(knots[0]), float(knots[1])
        self.__deg = deg
        self.__n = (knots.size-2)*(deg) + deg + 1
        self.__knots = jnp.array(knots)
        self.__coeffs = np.array([scipy.special.comb(deg, k) for k in range(deg+1)])
        super(PiecewiseBernsteinBasisJAX, self).__init__((a,b))
    
    @property
    def knots(self) -> jax.Array:
        """
        Return the knots as a `jax.Array`.

        Returns:
            jax.Array: the knots
        """
        return self.__knots
    
    @property
    def deg(self) -> int:
        """
        The degree of the Bernstein basis.

        Returns:
            int: degree.
        """
        return self.__deg
    
    @property
    def n(self) -> int:
        """
        The dimension of the Bernstein basis.

        Returns:
            int: the dimension.
        """
        return self.__n
    
    def __repr__(self) -> str:
        return 'Bernstein basis of degree '+str(self.__deg)+' and dimension '+str(self.__n)
    
    def _scaled_bernnstein_basis_eval(self, x: jax.Array, a: float, b: float, k: int, derivative: bool) -> jax.Array:
        
        x = (x-a)/(b-a)
        if derivative:
            if k == 0:
                return self.__coeffs[k]* ( - (self.__deg-k) * (x**k)*(1-x)**(self.__deg-k-1) )
            elif k == self.__deg:
                return self.__coeffs[k]* ( (k*x**(k-1))*(1-x)**(self.__deg-k) )
            else:
                return self.__coeffs[k]* ( (k*x**(k-1))*(1-x)**(self.__deg-k) - (self.__deg-k) * (x**k)*(1-x)**(self.__deg-k-1) )
        else:
            return self.__coeffs[k]*(x**k)*(1-x)**(self.__deg-k)
    
    
    def _eval_basis(self, x: jax.Array) -> jax.Array:
        
        result = jnp.zeros(x.shape+(self.__n,))
        
        for i in range(self.__knots.size-1):
            for k in range(self.__deg+1):
                tmp1 = jnp.where(jnp.logical_and(x>=self.__knots[i], x<=self.__knots[i+1]),self._scaled_bernnstein_basis_eval(x, self.__knots[i], self.__knots[i+1], k, False),0.0)
                tmp2 = jnp.where(jnp.logical_and(x>=self.__knots[i], x< self.__knots[i+1]),self._scaled_bernnstein_basis_eval(x, self.__knots[i], self.__knots[i+1], k, False),0.0)
                result_new = jnp.where(i == self.__knots.size-2, tmp1, tmp2)
                # if self.__knots[-1]<=self.__knots[i+1] and self.__knots[i]<self.__knots[-1]:
                #     idx = jnp.logical_and(x>=self.__knots[i], x<=self.__knots[i+1]) 
                # else:
                #     idx = jnp.logical_and(x>=self.__knots[i], x<self.__knots[i+1])
                result = result.at[..., (i+k//self.__deg)*self.__deg+k%self.__deg].set(result_new + result[...,(i+k//self.__deg)*self.__deg+k%self.__deg])
            
        return result.T
    
    def _eval_basis_derivative(self, x: jax.Array) -> jax.Array:
        
        result = jnp.zeros(x.shape+(self.__n,))
        
        for i in range(self.__knots.size-1):
            for k in range(self.__deg+1):
                tmp1 = jnp.where(jnp.logical_and(x>=self.__knots[i], x<=self.__knots[i+1]),self._scaled_bernnstein_basis_eval(x, self.__knots[i], self.__knots[i+1], k, True),0.0)
                tmp2 = jnp.where(jnp.logical_and(x>=self.__knots[i], x< self.__knots[i+1]),self._scaled_bernnstein_basis_eval(x, self.__knots[i], self.__knots[i+1], k, True),0.0)
                result_new = jnp.where(i == self.__knots.size-2, tmp1, tmp2)
                # if self.__knots[-1]<=self.__knots[i+1] and self.__knots[i]<self.__knots[-1]:
                #     idx = jnp.logical_and(x>=self.__knots[i], x<=self.__knots[i+1]) 
                # else:
                #     idx = jnp.logical_and(x>=self.__knots[i], x<self.__knots[i+1])
                result = result.at[..., (i+k//self.__deg)*self.__deg+k%self.__deg].set(result_new + result[...,(i+k//self.__deg)*self.__deg+k%self.__deg])
            
        return result.T
    
    def __call__(self, x : jax.Array|np.ndarray, derivative = False) -> jax.Array:
        """
        Evaluate the B-splines for the given points.

        Args:
            x (jax.Array | np.ndarray): the points where the basis is evaluated. Has to be vector of shape `(m,)`.
            derivative (bool, optional): evaluete the basis or its derivative. Defaults to False.

        Returns:
            jax.Array: the B-splines evaluated for x. Has the shape `(n,m)`, where `n` is the dimension of the basis.
        """
        if derivative:
            return self._eval_basis_derivative(jnp.array(x))
        else:
            return self._eval_basis(jnp.array(x))
        
    def interpolating_points(self) -> tuple[jax.Array, jax.Array]:
        pass 
    
    def quadrature_points(self,deg: int  = 1) -> tuple[jax.Array, jax.Array]:
        pts = []
        ws = []
        Pts, Ws = np.polynomial.legendre.leggauss(deg)
        for k in range(self.__knots.size-1):
            if self.__knots[k+1]>self.__knots[k]:
                pts += list(self.__knots[k]+(Pts+1)*0.5*(self.__knots[k+1]-self.__knots[k]))
                ws += list(Ws*(self.__knots[k+1]-self.__knots[k])/2)
        pts = np.array(pts)
        ws = np.array(ws)
        
        return pts, ws
        
    
class BSplineBasisJAX(UnivariateFunctionBasis):
    
    __deg : int 
    __n: int 
    __knots: jax.Array
    
    def __init__(self, knots: np.ndarray, deg: int, padded: bool = False):
        """_summary_

        Args:
            knots (np.ndarray): the knots of the basis (with/without padding).
            deg (int): the degree of the basis.
            padded (bool): the knot vector is padded or not.
        """
        
        self.__deg = deg
        self.__n = knots.size+deg-1
        if padded:
            self.__knots = jnp.array(knots)
        else:
            self.__knots = jnp.array(np.concatenate(( np.ones(deg)*knots[0], knots , np.ones(deg)*knots[-1] )))
        super(BSplineBasisJAX, self).__init__((float(knots[0]), float(knots[-1])))

    
    @property
    def knots(self) -> jax.Array:
        """
        Return the knots as a `jax.Array`.

        Returns:
            jax.Array: the knots
        """
        return self.__knots
    
    @property
    def deg(self) -> int:
        """
        The degree of the B-spline basis.

        Returns:
            int: degree.
        """
        return self.__deg
    
    @property
    def n(self) -> int:
        """
        The dimension of the B-spline basis.

        Returns:
            int: _description_
        """
        return self.__n
    
    def __repr__(self) -> str:
        return 'B-Spline basis of degree '+str(self.__deg)+' and dimension '+str(self.__n)
    
    def _eval_basis(self, x):
        x = jnp.array(x)
        result = jnp.zeros((x.shape[0],self.__knots.size-1))
        
        for i in range(self.__knots.size-1):

            
            tmp1 = jnp.where(jnp.logical_and(x>=self.__knots[i], x<=self.__knots[i+1]),1.0,0.0)
            tmp2 = jnp.where(jnp.logical_and(x>=self.__knots[i], x< self.__knots[i+1]),1.0,0.0)
            result_new = jnp.where(i==self.__n-1, tmp1, tmp2)
            # if self.__knots[-1]<=self.__knots[i+1] and self.__knots[i]<self.__knots[-1]:
            #     idx = jnp.logical_and(x>=self.__knots[i], x<=self.__knots[i+1]) 
            # else:
            #     idx = jnp.logical_and(x>=self.__knots[i], x<self.__knots[i+1])
            result = result.at[:,i].set(result_new)
            # result[idx,i] = 1.0
        
        for d in range(self.__deg):
            for i in range(result.shape[1]-d-1):
                a = result[:,i]*(x-self.__knots[i])/(self.__knots[i+d+1]-self.__knots[i])
                b = result[:,i+1]*(self.__knots[i+d+2]-x)/(self.__knots[i+d+2]-self.__knots[i+1])
                result_new = result.copy()
                result_new = result_new.at[:,i].set(0.0)
                # result_new[:,i] = 0.0
                
                result_new = jax.lax.cond((self.__knots[i+d+1]-self.__knots[i]) != 0, lambda : result_new.at[:,i].set( result_new[:,i] + a ), lambda : result_new)
                result_new = jax.lax.cond((self.__knots[i+d+2]-self.__knots[i+1])!=0, lambda : result_new.at[:,i].set( result_new[:,i] + b ), lambda : result_new)
                
                # if (self.__knots[i+d+1]-self.__knots[i]) != 0:
                #     result_new = result_new.at[:,i].set( result_new[:,i] + a )
                #     # result_new[:,i] = result_new[:,i] + a
                # if (self.__knots[i+d+2]-self.__knots[i+1])!=0:
                #     result_new = result_new.at[:,i].set( result_new[:,i] + b )
                #     # result_new[:,i] = result_new[:,i] + b
                result = result_new.copy() 
        # result[x==self.knots[-1],self.n-1] = 1.0
        return result[:,:self.__n].T
           
    def _eval_basis_derivative(self,x):

        x = jnp.array(x)
        
        result = jnp.zeros((x.shape[0],self.__knots.size-1))
        
        for i in range(result.shape[1]):
            tmp1 = jnp.where(jnp.logical_and(x>=self.__knots[i], x<=self.__knots[i+1]),1.0,0.0)
            tmp2 = jnp.where(jnp.logical_and(x>=self.__knots[i], x< self.__knots[i+1]),1.0,0.0)
            result_new = jnp.where(i==self.__n-1, tmp1, tmp2)
            # if self.__knots[-1]<=self.__knots[i+1] and self.__knots[i]<self.__knots[-1]:
            #     idx = jnp.logical_and(x>=self.__knots[i], x<=self.__knots[i+1]) 
            # else:
            #     idx = jnp.logical_and(x>=self.__knots[i], x<self.__knots[i+1])
            result = result.at[:,i].set(result_new)
            
            #    idx = np.logical_and(x>=self.__knots[i], x<=self.__knots[i+1]) if self.__knots[-1]<=self.__knots[i+1] and self.__knots[i]<self.__knots[-1] else jnp.logical_and(x>=self.__knots[i], x<self.__knots[i+1])
            #    # result = result.at[idx,i].set(1.0)
            #    result[idx,i] = 1.0
        #result[x==self.knots[-1],-1] = 1.0
        
        for d in range(self.__deg):
            for i in range(result.shape[1]-d-1):
                if d == self.__deg-1:
                    a = self.__deg*result[:,i]*(1)/(self.__knots[i+d+1]-self.__knots[i])
                    b = self.__deg*result[:,i+1]*(-1)/(self.__knots[i+d+2]-self.__knots[i+1])
                else:
                    a = result[:,i]*(x-self.__knots[i])/(self.__knots[i+d+1]-self.__knots[i])
                    b = result[:,i+1]*(self.__knots[i+d+2]-x)/(self.__knots[i+d+2]-self.__knots[i+1])
                result_new = result.copy()
                result_new = result_new.at[:,i].set(0.0)
                # result_new[:,i] = 0.0

                #   if (self.__knots[i+d+1]-self.__knots[i]) != 0:
                #       result_new  = result_new.at[:,i].set(result_new[:,i] + a)
                #       # result_new[:,i] = result_new[:,i] + a
                #   if (self.__knots[i+d+2]-self.__knots[i+1])!=0:
                #       result_new = result_new.at[:,i].set(result_new[:,i] + b)
                #       # result_new[:,i] = result_new[:,i] + b

                result_new = jax.lax.cond((self.__knots[i+d+1]-self.__knots[i]) != 0, lambda : result_new.at[:,i].set( result_new[:,i] + a ), lambda : result_new)
                result_new = jax.lax.cond((self.__knots[i+d+2]-self.__knots[i+1])!=0, lambda : result_new.at[:,i].set( result_new[:,i] + b ), lambda : result_new)

                result = result_new.copy() 
                
        # result[x==self.knots[-1],self.n-1] = 1.0
        return result[:,:self.__n].T
            
    def __call__(self, x : jax.Array|np.ndarray, derivative = False) -> jax.Array:
        """
        Evaluate the B-splines for the given points.

        Args:
            x (jax.Array | np.ndarray): the points where the basis is evaluated. Has to be vector of shape `(m,)`.
            derivative (bool, optional): evaluete the basis or its derivative. Defaults to False.

        Returns:
            jax.Array: the B-splines evaluated for x. Has the shape `(n,m)`, where `n` is the dimension of the basis.
        """
        if derivative:
            return self._eval_basis_derivative(jnp.array(x))
        else:
            return self._eval_basis(jnp.array(x))
   
    def interpolating_points(self) -> tuple[jax.Array, jax.Array]:
        """
        Return the intepolating points and the basis evaluated in these points.
        The resulting matrix is nonsingular.
        
        Example:
        ```
        
        ```


        Returns:
            tuple[jax.Array, jax.Array]: the points as a vector and the matrix resulted from evaluating the basis for these points. 
        """
        pts = jnp.array([jnp.sum(self.__knots[i+1:i+self.__deg+1]) for i in range(self.n)])/(self.__deg)
        Mat = self.__call__(pts)
        return pts, Mat
    
    def quadrature_points(self,deg: int  = 1) -> tuple[jax.Array, jax.Array]:
        pts = []
        ws = []
        Pts, Ws = np.polynomial.legendre.leggauss(deg)
        for k in range(self.__knots.size-1):
            if self.__knots[k+1]>self.__knots[k]:
                pts += list(self.__knots[k]+(Pts+1)*0.5*(self.__knots[k+1]-self.__knots[k]))
                ws += list(Ws*(self.__knots[k+1]-self.__knots[k])/2)
        pts = np.array(pts)
        ws = np.array(ws)
        
        return pts, ws