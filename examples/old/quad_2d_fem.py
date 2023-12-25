import fenics as fn
import matplotlib.pyplot as plt
from dolfin_utils.meshconvert import meshconvert
import os
from subprocess import call
import numpy as np

def curl2D(v):
    return fn.as_vector((v.dx(1),-v.dx(0)))

class FEM():
            
    def __init__(self, J0=14e6, mu0=4*np.pi*1e-7, mur=2000.0, k1=1, k2=1.65, k3 = 500, meshsize = 0.001, params = [0,0,0,0,0,0], nonlin = True, verb = False):
        # scale_H = 1000
        # scale_A = 70
        
        # k1 = k1/scale_H
        # k2 = k2/(scale_A**2)
        # k3 = k3/scale_H
        # mu0 = mu0*scale_H
        # J0 = J0 / (scale_H / scale_A)
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        fn.set_log_active(verb)
        path='./quad/'
        
        with open(path + 'quad_param.geo', 'r') as file:
            data = file.read()
            
        s = "meshsize=%.18f;\np0 = %.18f;\np1 = %.18f;\np2 = %.18f;\n"%(meshsize,params[0],params[1],params[2])
        
        s = s + data
        
        with  open(path+"tmp.geo", "w") as file:
            file.write(s)
            file.close()
        if verb: print('geo file created',flush = True)
        
        if verb:
            os.system('gmsh %stmp.geo -nt 20 -3 -o %stmp.msh -format msh2 '%(path,path))
        else:
            os.system('gmsh %stmp.geo -nt 20 -3 -o %stmp.msh -format msh2 >/dev/null 2>&1'%(path,path))
        if verb: print('mesh file created',flush=True)

        if verb:
            os.system('dolfin-convert %stmp.msh %stmp.xml'%(path,path))
        else:
            os.system('dolfin-convert %stmp.msh %stmp.xml >/dev/null 2>&1'%(path,path))
        
        if verb: print('mesh file converted in fenics format',flush=True) 

        mesh = fn.Mesh(path+'tmp.xml')
        domains = fn.MeshFunction("size_t", mesh, path+'tmp_physical_region.xml')
        boundaries = fn.MeshFunction('size_t', mesh, path+'tmp_facet_region.xml')

        self.mesh = mesh
        ncells = [  mesh.num_vertices(), mesh.num_edges(), mesh.num_faces(), mesh.num_facets(), mesh.num_cells() ]
        
        def nu_lin(az):
            return 1/(mu0*mur)

        def nonlin_nu(az):
            tmp =  (k1*(1+0.1*params[3]))*fn.exp((k2*(1+0.1*params[4]))*fn.dot(az.dx(1),az.dx(1)))+(k3*(1+0.1*params[5]))
            return tmp
        
        def nu_Bauer(B):
            x = fn.dot(B,B)
            return (k1*(1+0.1*params[3]))*fn.exp((k2*(1+0.1*params[4]))*x)+(k3*(1+0.1*params[5]))
        
        # Coil
        def setup_coil(mesh,subdomains):
            DG = fn.FunctionSpace(mesh,"DG",0)
            J = fn.Function(DG)
            idx = []
            for cell_no in range(len(subdomains.array())):
                subdomain_no = subdomains.array()[cell_no]
                if subdomain_no == 3:
                    idx.append(cell_no)
            J.vector()[:] = 0
            J.vector()[idx] = J0
            return J
        
    
        
        """ define function space and boundary conditions"""
        
        CG = fn.FunctionSpace(mesh, 'CG', 1) # Continuous Galerkin
        
        # Define boundary condition
        bc = fn.DirichletBC(CG, fn.Constant(0.0), boundaries,16)
        
        # Define subdomain markers and integration measure
        dx = fn.Measure('dx', domain=mesh, subdomain_data=domains)
        
        J = setup_coil(mesh, domains)
        
        class Nu(fn.UserExpression): # UserExpression instead of Expression
            def __init__(self, markers, **kwargs):
                super().__init__(**kwargs) # This part is new!
                self.markers = markers
            def eval_cell(self, values, x, cell):
                if self.markers[cell.index] == 1:
                    values[0] = 0.0   # iron
                elif self.markers[cell.index] == 2:
                    values[0] = 1/mu0      # air
                elif self.markers[cell.index] == 3:
                    values[0] = 1/mu0      # air
                else:
                    print('no such domain',self.markers[cell.index] )
                    
        nus = Nu(domains, degree=1)
        
        
        """ weak formulation """
        
        az  = fn.Function(CG)
        u  = fn.Function(CG)
        v  = fn.TestFunction(CG)
        #az = Function(CG)
        #a  = (1/mu)*dot(grad(az), grad(v))*dx
        if nonlin:
            a = fn.inner(nus*curl2D(u), curl2D(v))*dx + fn.inner(nu_Bauer(curl2D(u))*curl2D(u),curl2D(v))*dx(1)
        else:
            a = fn.inner(nus*curl2D(u), curl2D(v))*dx + fn.inner(nu_lin(curl2D(u))*curl2D(u),curl2D(v))*dx(1)

        L  = J*v*dx(3)
        
        F = a - L
        # solve variational problem
        fn.solve(F == 0, u, bc)
        az = u
        self.az = az
        # function space for H- and B- field allocated on faces of elements
        W = fn.VectorFunctionSpace(mesh, fn.FiniteElement("DP", fn.triangle, 0),1)
        B = fn.project(curl2D(az), W)
        H = None# project((1/mu)*curl(az), W)
        self.B = B
        self.H = H
        self.surface_cu = fn.assemble(fn.Constant(1.0)*dx(3))
    
    @property
    def u(self):
        return self.az
    
    def __call__(self,xs):
        
        
        Afem = 0 * xs[:,0]
        for i in range(xs.shape[0]):
            try:
                Afem[i] = self.az(xs[i,0],xs[i,1])
            except:
                Afem[i] = np.nan
        return Afem
    
    def call_B(self,xs):
        
        
        Bfem = xs*0
        for i in range(xs.shape[0]):
            try:
                Bfem[i,:] = self.B(xs[i,0],xs[i,1])
            except:
                Bfem[i,:] = np.nan
        return np.array(Bfem)
    
    def call_H(self,x_eval,y_eval):
        Hfem = []
        for i in range(x_eval.size):
            try:
                Hfem.append(self.H(x_eval[i],y_eval[i]))
            except:
                Hfem.append([ np.nan , np.nan])
        return np.array(Hfem)
    
    
if __name__ == '__main__':
    
    fem_solver = FEM(verb = True)
    
    xs = np.meshgrid(np.linspace(0,0.075,200), np.linspace(0,0.055,200))
    
    az = fem_solver(np.concatenate((xs[0].flatten()[...,None], xs[1].flatten()[...,None]),-1)).reshape(xs[0].shape)
    
    plt.figure()
    plt.contourf(xs[0], xs[1], az, levels=32)
    plt.xlabel(r'$x_1$ [m]')
    plt.ylabel(r'$x_2$ [m]') 
    plt.colorbar()
    

    xs = np.meshgrid(np.linspace(0,0.075,100), np.linspace(0,0.055,100))
    B = fem_solver.call_B(np.concatenate((xs[0].flatten()[...,None], xs[1].flatten()[...,None]),-1))
    
    plt.figure()
    plt.quiver(xs[0], xs[1], B[:,0], B[:,1])
    plt.xlabel(r'$x_1$ [m]')
    plt.ylabel(r'$x_2$ [m]') 
    plt.colorbar()
    
    Bnorm = np.linalg.norm(B,axis=-1)

    Hmax = (fem_solver.k1*np.exp(fem_solver.k2*np.nanmax(Bnorm)**2)+fem_solver.k3)*np.nanmax(Bnorm)
    
    print('Maximum B ',np.nanmax(Bnorm))
    print('Maximum H', Hmax)