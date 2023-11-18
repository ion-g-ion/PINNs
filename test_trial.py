import jax
import jax.numpy as jnp
from jax.example_libraries import stax, optimizers
import numpy as np
import matplotlib.pyplot as plt
import pinns 
import datetime
import pytest
import pyvista 

init, nn = stax.serial(stax.Dense(8), stax.Relu, stax.Dense(10), stax.Relu, stax.Dense(1))
ws = init(jax.random.PRNGKey(12234534), (3,))[1]
pinn = pinns.FunctionNN(nn, init, ((-1,1), (-2,1), (0, 1)))

x1, x2, x3 = np.meshgrid(np.linspace(-1,1,64), np.linspace(-2,1,63), np.linspace(0,1,65))
x_in = np.concatenate((x1.reshape([-1,1], order='F'), x2.reshape([-1,1], order='F'), x3.reshape([-1,1], order='F')), 1)

vals = pinn(ws, x_in)

obj1 = pyvista.StructuredGrid(x1, x2, x3).cast_to_unstructured_grid()
obj1.point_data['value'] = vals.flatten(order='F')

facefun = pinn.face_function(1,0,1,-1,((-1,1),(-1,1),(0,1)))

#%% object 2
x1, x2, x3 = np.meshgrid(np.linspace(-1,1,64), np.linspace(-1,1,63), np.linspace(0,1,65))
x_in = np.concatenate((x1.reshape([-1,1], order='F'), x2.reshape([-1,1], order='F'), x3.reshape([-1,1], order='F')), 1)
vals = facefun(ws, x_in)

obj2 = pyvista.StructuredGrid(x1, x2-3, x3).cast_to_unstructured_grid()
obj2.point_data['value'] = vals.flatten(order='F')

#%% object 3 
facefun = pinn.interface_function((0,1),(0,0),(0,1),(-1,0),((0,1),(-1,1),(0,1)))

x1, x2, x3 = np.meshgrid(np.linspace(0,1,64), np.linspace(-1,1,63), np.linspace(0,1,65))
x_in = np.concatenate((x1.reshape([-1,1], order='F'), x2.reshape([-1,1], order='F'), x3.reshape([-1,1], order='F')), 1)
vals = facefun(ws, x_in)

obj3 = pyvista.StructuredGrid(-x2-2, x1*2-4, x3).cast_to_unstructured_grid()
obj3.point_data['value'] = vals.flatten(order='F')

#%% plot 
obj_total = obj1.merge(obj2)
obj_total = obj_total.merge(obj3)
obj_total.save('testing.vtk')

pl = pyvista.Plotter()
_ = pl.add_mesh(obj1, cmap='jet', show_edges=True)
_ = pl.add_mesh(obj2, cmap='jet')
_ = pl.add_mesh(obj3, cmap='jet')
pl.add_bounding_box()
pl.show_grid()
pl.show()
