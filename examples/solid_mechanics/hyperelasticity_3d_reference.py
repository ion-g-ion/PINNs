import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime 
import pyvista as pv

fe.parameters["allow_extrapolation"] = True

class FEM:
    
    def __init__(self, E =  0.02e5, nu = 0.1, rho = 0.4, g = 9.81, params = [0,0,0,0,0,0]):
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
        

        


        # --------------------
        # Function spaces
        # --------------------
        V = fe.VectorFunctionSpace(mesh, "CG", 1)
        self.V = V
        u_tr = fe.TrialFunction(V)
        u_test = fe.TestFunction(V)
        u = fe.Function(V)
        # g = fe.Constant((0.0, g_int))
        b = fe.Constant((0.0, 0.0, -self.__g*self.__rho))
        #N = fe.Constant((0.0, 1.0))

        dx = fe.Measure('dx', domain=mesh, subdomain_data=domains)
        # --------------------
        # Boundary conditions
        # --------------------
        bc = fe.DirichletBC(V, fe.Constant((0.0, 0.0, 0.0)), boundaries, 152)


        aa, bb, cc, dd, ee = 0.5*mu, 0.0, 0.0, mu, -1.5*mu

        # --------------------
        # Weak form
        # --------------------
        I = fe.Identity(3)
        F = I + fe.grad(u)  # Deformation gradient
        C = F.T*F  # Right Cauchy-Green tensor
        J = fe.det(F)  # Determinant of deformation fradient

        #psi = (aa*fe.tr(C) + bb*fe.tr(ufl.cofac(C)) + cc*J**2 - dd*fe.ln(J))*fe.dx - fe.dot(b, u)*fe.dx + fe.inner(f, u)*ds(1)
        #n = fe.dot(ufl.cofac(F), N)
        #surface_def = fe.sqrt(fe.inner(n, n))
        psi = (aa*fe.inner(F, F) + ee - dd*fe.ln(J))*fe.dx - J*fe.dot(b, u)*fe.dx  # + surface_def*fe.inner(g, u)*ds(1)
        psi = (0.5*mu*(fe.tr(C)-3)-mu*fe.ln(J)+0.5*lambda_*fe.ln(J)**2)*fe.dx - J*fe.dot(b, u)*fe.dx 
        # --------------------
        # Solver
        # --------------------
        Form = fe.derivative(psi, u, u_test)
        Jac = fe.derivative(Form, u, u_tr)

        problem = fe.NonlinearVariationalProblem(Form, u, bc, Jac)
        solver = fe.NonlinearVariationalSolver(problem)

        
        tme_current = datetime.datetime.now()

        prm = solver.parameters
        #prm["newton_solver"]["error_on_convergence"] = False
        #fe.solve(Form == 0, u, bc, J=Jac, solver_parameters={"error_on_convergence": False})
        solver.solve()

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

    obj = pv.read('solution.vtk')
    dis = fem(np.array(obj.points))
    obj.point_data['displacement_fenics'] = dis
    obj.save('solution_new.vtk')
    #u_topology, u_cell_types, u_geometry = fe.plot.create_vtk_mesh(fem.V)
    #grid = pv.UnstructuredGrid(fem.mesh.cells(), types, x)
    #grid.point_data["u"] = u.x.array