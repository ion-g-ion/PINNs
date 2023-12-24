import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime 
import plotly.figure_factory as ff
import plotly.express as px

class FEM:
    
    def __init__(self, E =  0.02e5, nu = 0.1, rho = 0.1, g = 9.81, params = [0,0,0,0,0,0]):
        self.__E = E
        self.__nu = nu
        self.__rho = rho
        self.__g = g
        self.__solved = False
        self.__u = None
        self.__params = params 
        
    @property
    def u(self):
        return self.__u

    @property 
    def mesh(self):
        return self.__mesh
    
    def solve(self, meshsize = 0.1, verb = False):
        
        # Mesh creation
        path='./'
        
        with open(path + 'elastostatics_3d.geo', 'r') as file:
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
            return lambda_*fe.div(u)*fe.Identity(3) + 2*mu*epsilon(u)
        

        
        b = fe.Constant((0.0, 0.0, -self.__g*self.__rho))

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

        dx = fe.Measure('dx', domain=mesh, subdomain_data=domains)
        # --------------------
        # Boundary conditions
        # --------------------
        bc = fe.DirichletBC(V, fe.Constant((0.0, 0.0, 0.0)), boundaries, 152)

        # --------------------
        # Weak form
        # --------------------
        a = fe.inner(sigma(u_tr), epsilon(u_test))*dx
        l = fe.dot(b, u_test)*dx # + fe.inner(g, u_test)*ds(1)

        # --------------------
        # Solver
        # --------------------
        u = fe.Function(V)
        if verb: print('Assembling the system...',flush=True)
        A_ass, L_ass = fe.assemble_system(a, l, bc)

        if verb: print('Solving the system...',flush=True)
        
        # solver = fe.KrylovSolver("cg", "ml_amg")
        # solver.parameters["relative_tolerance"] = 5e-6
        # solver.parameters["maximum_iterations"] = 1000
        # solver.parameters["monitor_convergence"] = True


        

        tme_current = datetime.datetime.now()
        
        solver = fe.KrylovSolver("cg")
        solver.parameters["relative_tolerance"] = 1e-10
        solver.parameters["maximum_iterations"] = 10000
        solver.parameters["monitor_convergence"] = False
        solver.solve(A_ass, u.vector(), L_ass)


        # fe.solve(A_ass, u.vector(), L_ass,"gmres")
        # possible solvers "mumps", "cg", "petsc"
        
        # fe.solve(A_ass, u.vector(), L_ass, solver_parameters={"linear_solver": "cg", "relative_tolerance": 1e-6}, form_compiler_parameters={"optimize": True})
        tme_current = datetime.datetime.now() - tme_current

        if verb: print('\truntime ',tme_current)
        self.__u = u
        
        if verb: print('System solved!!!')
        # print('Volume ',fe.assemble(fe.Constant(1.0)*dx))
        
        self.__solved = True

    def __call__(self, xs):
        if not self.__solved:
            self.solve()
            
        dis = 0 * xs
        for i in range(xs.shape[0]):
            try:
                dis[i,:] = self.__u(xs[i,0],xs[i,1],xs[i,2])
            except:
                dis[i,:] = np.nan
        return dis
    
    
    
    
if __name__ == "__main__":
    fem = FEM()
    fem.solve(0.1, False)
    
    
    
    xs = np.meshgrid(np.linspace(-1,3,32), np.linspace(-3,3,32), np.linspace(0,2,32))
    
    dis = fem(np.concatenate((xs[0].flatten()[...,None],xs[1].flatten()[...,None],xs[2].flatten()[...,None]),-1))
    color = np.sqrt(dis[:,0] + dis[:,1] + dis[:,2])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.quiver(xs[0].flatten(), xs[1].flatten(), xs[2].flatten(), dis[:,0], dis[:,1], dis[:,2])

    print('DoFs %d'%(fem.u.vector().size()))
    # plt.figure()
    # plt.quiver(xs[0].flatten(), xs[1].flatten(), dis[:,0], dis[:,1])
    
    # plt.figure()
    # plt.scatter(xs[0].flatten()+dis[:,0], xs[1].flatten()+dis[:,1],s=1)
    
    # fe.plot(fem.u, 
    #        mode="displaced mesh",
    #        lighting='plastic',
    #        axes=1,
    #        viewup='z',
    #        interactive=0)
    # fe.plot(fem.mesh)
    # plt.show()


    px.scatter_3d(x=xs[0].flatten(), y=xs[1].flatten(), z=xs[2].flatten(), color=color)