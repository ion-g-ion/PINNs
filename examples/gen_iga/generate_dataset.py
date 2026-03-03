import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2
from pyiga import bspline, assemble, geometry, vis
from scipy.spatial.distance import cdist
from scipy.linalg import cho_solve, cho_factor
import scipy.sparse.linalg
import h5py
from tqdm import tqdm

def generate_geom(N=(128, 128), deg=2, epsilon_strain=4.0, epsilon_rotation=3.0):
    Nx, Ny = N  # Unpack N into Nx and Ny
    # Step 1: Grid
    x_max = np.random.uniform(0.5, 2.5)
    y_max = np.random.uniform(0.5, 2.5)
    x = np.linspace(0, x_max, Nx)
    y = np.linspace(0, y_max, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    J = 0.0
    while J < 0.08:
        
        # Step 2: Generate symmetric (strain) and antisymmetric (rotation) fields
        h_xx = epsilon_strain * gaussian_filter(np.random.randn(Nx, Ny), sigma=5) 
        h_yy = epsilon_strain * gaussian_filter(np.random.randn(Nx, Ny), sigma=5)
        h_xy = epsilon_strain * gaussian_filter(np.random.randn(Nx, Ny), sigma=5)

        rotation_field = epsilon_rotation * gaussian_filter(np.random.randn(Nx, Ny), sigma=8)
        rotation_field = epsilon_rotation * np.random.randn() * np.exp(-((X-np.random.rand()*x.max())**2 + (Y-np.random.rand()*y.max())**2)/0.5)

        # Step 3: Build divergence terms
        div_hx = np.gradient(h_xx, x, axis=0) + np.gradient(h_xy, y, axis=1)
        div_hy = np.gradient(h_xy, x, axis=0) + np.gradient(h_yy, y, axis=1)

        div_hx += np.gradient(rotation_field, y, axis=1)
        div_hy += -np.gradient(rotation_field, x, axis=0)

        # Step 4: Fourier solve
        kx = np.fft.fftfreq(Nx, d=x[1]-x[0]) * 2*np.pi
        ky = np.fft.fftfreq(Ny, d=y[1]-y[0]) * 2*np.pi
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        Laplacian = -(KX**2 + KY**2)
        Laplacian[0, 0] = 1.0

        div_hx_hat = fft2(div_hx)
        div_hy_hat = fft2(div_hy)

        u_x_hat = div_hx_hat / Laplacian
        u_y_hat = div_hy_hat / Laplacian

        u_x = np.real(ifft2(u_x_hat))
        u_y = np.real(ifft2(u_y_hat))

        # Step 5: New grid
        X_new = X + u_x
        Y_new = Y + u_y

        # Center around the center of mass
        X_com = np.mean(X_new)
        Y_com = np.mean(Y_new)
        X_new -= X_com
        Y_new -= Y_com

        # Apply random rotation
        theta = np.random.uniform(0, 2 * np.pi)  # Random rotation angle
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                    [np.sin(theta), np.cos(theta)]])
        coords = np.stack([X_new.ravel(), Y_new.ravel()])
        rotated_coords = rotation_matrix @ coords
        X_new = rotated_coords[0].reshape(X_new.shape)
        Y_new = rotated_coords[1].reshape(Y_new.shape)

        # define discretization space
        kvs = tuple(bspline.make_knots(deg, 0.0, 1.0, n-deg) for n in N)

        geo = geometry.tensor_product(geometry.BSplineFunc(kvs[0], np.linspace(0, 1,kvs[0].numdofs)), geometry.BSplineFunc(kvs[1], np.linspace(0,1,kvs[1].numdofs)))
        # Update geo.coeffs with new coordinates
        geo.coeffs = np.stack([Y_new, X_new], axis=-1)

        J = geo.grid_jacobian([s.greville()[1:-1] for s in kvs])
        J = np.linalg.det(J).min() / np.linalg.det(J).max()
        #print("Jacobian determinant ratio:", J)
        
    return geo, kvs, X_new, Y_new

def generate_bcs(geo, kvs, scale = 2.0, sigma=16.0):
    # Define boundary conditions

    g = lambda x,y: 0*x # source term
    # compute Dirichlet boundary conditions
    bcs = assemble.compute_dirichlet_bcs(kvs, geo, [('left', g), ('top', g), ('right', g), ('bottom', g)])
    
    V = scale * np.random.randn() * gaussian_filter(np.random.randn(*geo.coeffs.shape[:2]), sigma=sigma).flatten()
    
    bcs[1][:] = V[bcs[0]]
    return bcs





# Example usage
if __name__ == "__main__":
    n_samples = 10000
    N = (32, 32)
    deg = 2
    output_file = "generated_data_small.h5"

    with h5py.File(output_file, "w") as h5f:
        solutions_dataset = h5f.create_dataset("solutions", (n_samples, np.prod(N)), dtype=np.float64)
        geo_coeffs_dataset = h5f.create_dataset("geo_coeffs", (n_samples, 2 * np.prod(N)), dtype=np.float64)

        for k in tqdm(range(n_samples), desc="Generating samples"):
            geo, spline_space, X_new, Y_new = generate_geom(N, deg)
            bcs = generate_bcs(geo, spline_space)
            
            rhs = assemble.inner_products(spline_space, lambda x, y: 0 * x, f_physical=True, geo=geo).ravel()
            A = assemble.stiffness(spline_space, geo)
            LS = assemble.RestrictedLinearSystem(A, rhs, bcs)

            u = scipy.sparse.linalg.spsolve(LS.A, LS.b)
            u_func = geometry.BSplineFunc(spline_space, LS.complete(u))

            # Create datasets dynamically in the first iteration when sizes are known
            if k == 0:
                mats_data_dataset = h5f.create_dataset("mats_data", (n_samples, LS.A.data.shape[0]), dtype=np.float64)
                mats_indices_dataset = h5f.create_dataset("mats_indices", (n_samples, LS.A.indices.shape[0]), dtype=np.int32)
                mats_indptr_dataset = h5f.create_dataset("mats_indptr", (n_samples, LS.A.indptr.shape[0]), dtype=np.int32)
                rhs_dataset = h5f.create_dataset("rhs", (n_samples, LS.b.shape[0]), dtype=np.float64)

            # Save data to HDF5
            solutions_dataset[k] = u_func.coeffs.flatten()
            mats_data_dataset[k, :LS.A.data.shape[0]] = LS.A.data
            mats_indices_dataset[k, :LS.A.indices.shape[0]] = LS.A.indices
            mats_indptr_dataset[k, :LS.A.indptr.shape[0]] = LS.A.indptr
            rhs_dataset[k, :LS.b.shape[0]] = LS.b
            geo_coeffs_dataset[k] = geo.coeffs.flatten()

        # Ensure data is written to disk
        h5f.flush()

    plt.figure()
    vis.plot_geo(geo, grid = 32)
    plt.savefig("generated_geometry.jpg")
    
    plt.figure()
    vis.plot_field(u_func, geo)
    plt.savefig("generated_field.jpg")
