import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
import os

class FEM:
    
    def __init__(self, E = 2.0e7, nu = 0.0, rho = 200.0, g = 9.81):
        self.__E = E
        self.__nu = nu
        self.__rho = rho
        self.__g = g
        self.__solved = False
        self.__u = None
        
    @property
    def u(self):
        return self.__u

    @property 
    def mesh(self):
        return self.__mesh
    
    def solve(self, meshsize = 0.025, verb = False):
        
        # Mesh creation
        path='./'
        
        with open(path + 'quarter_circle.geo', 'r') as file:
            data = file.read()
            
        s = "meshsize=%.18f;\n"%(meshsize)
        
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

        mesh = fe.Mesh(path+'tmp.xml')
        domains = fe.MeshFunction("size_t", mesh, path+'tmp_physical_region.xml')
        boundaries = fe.MeshFunction('size_t', mesh, path+'tmp_facet_region.xml')
        self.__mesh = mesh
        

     

        # Lame's constants
        lambda_ = self.__E*self.__nu/(1+self.__nu)/(1-2*self.__nu)
        mu = self.__E/2/(1+self.__nu)

        # Strain function
        def epsilon(u):
            return fe.sym(fe.grad(u))

        # Stress function
        def sigma(u):
            return lambda_*fe.div(u)*fe.Identity(2) + 2*mu*epsilon(u)
        

        # Load
        # g_z = -2.9575e5
       
        # g = fe.Constant((0.0, g_z))
        b = fe.Constant((0.0, -self.__g*self.__rho))

        # # Definition of Neumann condition domain
        # boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        # boundaries.set_all(0)

        # top = fe.AutoSubDomain(lambda x: fe.near(x[1], l_y))

        # top.mark(boundaries, 1)
        # ds = fe.ds(subdomain_data=boundaries)

        # --------------------
        # Function spaces
        # --------------------
        V = fe.VectorFunctionSpace(mesh, "CG", 1)
        u_tr = fe.TrialFunction(V)
        u_test = fe.TestFunction(V)

        # --------------------
        # Boundary conditions
        # --------------------
        bc = fe.DirichletBC(V, fe.Constant((0.0, 0.0)), boundaries, 6)

        # --------------------
        # Weak form
        # --------------------
        a = fe.inner(sigma(u_tr), epsilon(u_test))*fe.dx
        l = fe.dot(b, u_test)*fe.dx # + fe.inner(g, u_test)*ds(1)

        # --------------------
        # Solver
        # --------------------
        u = fe.Function(V)
        A_ass, L_ass = fe.assemble_system(a, l, bc)

        fe.solve(A_ass, u.vector(), L_ass)

        self.__u = u

    def __call__(self, xs):
        if not self.__solved:
            self.solve()
            
        dis = 0 * xs
        for i in range(xs.shape[0]):
            try:
                dis[i,:] = self.__u(xs[i,0],xs[i,1])
            except:
                dis[i,:] = np.nan
        return dis
    
    
    
    
if __name__ == "__main__":
    fem = FEM(E = 0.02e5, nu = 0.1, rho = 2)
    fem.solve()
    
    print(np.amax(fem.u.vector()[:]))

    fe.plot(fem.u, mode="displacement")
    
    
    xs = np.meshgrid(np.linspace(0,2.1,32), np.linspace(0,2.1,32))
    
    dis = fem(np.concatenate((xs[0].flatten()[...,None],xs[1].flatten()[...,None]),-1))
    
    plt.figure()
    plt.quiver(xs[0].flatten(), xs[1].flatten(), dis[:,0], dis[:,1])
    
    plt.figure()
    plt.scatter(xs[0].flatten()+dis[:,0], xs[1].flatten()+dis[:,1],s=1)
    
    plt.figure()
    fe.plot(fem.mesh)
    
    plt.show()