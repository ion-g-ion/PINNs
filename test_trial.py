import jax
import jax.numpy as jnp
from jax.example_libraries import stax, optimizers
import numpy as np
import matplotlib.pyplot as plt
import pinns 
import datetime
import pytest
import pyvista 

acti = stax.elementwise(lambda x: jnp.sin(2*x))
    
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

init, nn = stax.serial(stax.Dense(8), acti, stax.Dense(10), acti, stax.Dense(1))
pinn = pinns.FunctionSpaceNN((init, nn), ((-1,1), (-1,1)))
ws = pinn.init_weights(jax.random.PRNGKey(123456))

x1, x2= np.meshgrid(np.linspace(-1,1,64), np.linspace(-1,1,63))
x_in = np.concatenate((x1.reshape([-1,1], order='F'), x2.reshape([-1,1], order='F')), 1)
vals = pinn.neural_network(ws, x_in)
obj1 = pyvista.StructuredGrid(x1, x2, np.array(vals.reshape(x1.shape, order='F'))).cast_to_unstructured_grid()
obj1.point_data['value'] = vals.flatten(order='F')

facefun_top = pinn.interface_function((1,),(-1,),(1,),(0,),((-1,1),(0,2)), ((0,1),(1,1)))
x1, x2= np.meshgrid(np.linspace(-1,1,64), np.linspace(0,2,63))
x_in = np.concatenate((x1.reshape([-1,1], order='F'), x2.reshape([-1,1], order='F')), 1)
vals_top = facefun_top(ws, x_in)
obj2 = pyvista.StructuredGrid(x1, x2+1, np.array(vals_top.reshape(x1.shape, order='F'))).cast_to_unstructured_grid()
obj2.point_data['value'] = vals_top.flatten(order='F')

facefun_right = pinn.interface_function((0,),(-1,),(1,),(-1,),((-1,1),(0,1)), ((1,1),(0,-1)))
x1, x2= np.meshgrid(np.linspace(-1,1,64), np.linspace(0,1,63))
x_in = np.concatenate((x1.reshape([-1,1], order='F'), x2.reshape([-1,1], order='F')), 1)
vals_right = facefun_right(ws, x_in)
obj3 = pyvista.StructuredGrid(2-x2, x1, np.array(vals_right.reshape(x1.shape, order='F'))).cast_to_unstructured_grid()
obj3.point_data['value'] = vals_right.flatten(order='F')

facefun_tr = pinn.interface_function((0,1),(-1,-1),(0,1),(-1,0),((0,2),(0,1)), ((1,-1),(0,1)))
x1, x2= np.meshgrid(np.linspace(0,2,64), np.linspace(0,1,63))
x_in = np.concatenate((x1.reshape([-1,1], order='F'), x2.reshape([-1,1], order='F')), 1)
vals_tr = facefun_tr(ws, x_in)
#ax.plot_surface(1+x2, 3-x1, vals_tr.reshape(x1.shape, order='F'), vmin = vals.min(), vmax=vals.max())
obj4 = pyvista.StructuredGrid(1+x2, 3-x1, np.array(vals_tr.reshape(x1.shape, order='F'))).cast_to_unstructured_grid()
obj4.point_data['value'] = vals_tr.flatten(order='F')

#plt.show()

pl = pyvista.Plotter()
_ = pl.add_mesh(obj1, cmap='jet', show_edges=True)
_ = pl.add_mesh(obj2, cmap='jet', show_edges=True)
_ = pl.add_mesh(obj3, cmap='jet', show_edges=True)
_ = pl.add_mesh(obj4, cmap='jet', show_edges=True)

pl.add_bounding_box()
pl.show_grid()
pl.show()

ws = dict()
ws['lb'] = pinn.init_weights(jax.random.PRNGKey(123456))
ws['rb'] = pinn.init_weights(jax.random.PRNGKey(4324324))
ws['lt'] = pinn.init_weights(jax.random.PRNGKey(5686))
ws['rt'] = pinn.init_weights(jax.random.PRNGKey(257980))

conns = [
    pinns.PatchConnectivity(first='lb', second='rb', axis_first=(0,), axis_second=(1,), end_first=(-1,), end_second=(-1,), axis_permutation=((1,-1),(0,1))),
    pinns.PatchConnectivity(first='rt', second='rb', axis_first=(0,), axis_second=(0,), end_first=(-1,), end_second=(-1,), axis_permutation=((0,-1),(1,-1))),
    pinns.PatchConnectivity(first='lb', second='rt', axis_first=(0,1), axis_second=(0,1), end_first=(-1,-1), end_second=(-1,0), axis_permutation=((1,1),(0,-1)))
]

#ws['rb'][-1][1].at[0].set(0)
#ws['rb'][-1][0].at[:].set(0)
#ws['rt'][-1][1].at[0].set(0)
#ws['rt'][-1][0].at[:].set(0)

funcs = pinns.connectivity_to_interfaces({'lb': pinn, 'rb': pinn, 'rt': pinn}, conns)

x1, x2= np.meshgrid(np.linspace(-1,1,64), np.linspace(-1,1,63))
x_in = np.concatenate((x1.reshape([-1,1], order='F'), x2.reshape([-1,1], order='F')), 1)

vals = dict()
for key in funcs:
    vals[key] = funcs[key](ws, x_in)
    
obj1 = pyvista.StructuredGrid(x1, x2, np.array(vals['lb'].reshape(x1.shape, order='F'))).cast_to_unstructured_grid()
obj1.point_data['value'] = vals['lb'].flatten(order='F')
obj2 = pyvista.StructuredGrid(2-x2, x1, np.array(vals['rb'].reshape(x1.shape, order='F'))).cast_to_unstructured_grid()
obj2.point_data['value'] = vals['rb'].flatten(order='F')
obj3 = pyvista.StructuredGrid(2+x2, 2-x1, np.array(vals['rt'].reshape(x1.shape, order='F'))).cast_to_unstructured_grid()
obj3.point_data['value'] = vals['rt'].flatten(order='F')

pl = pyvista.Plotter()
_ = pl.add_mesh(obj1, cmap='jet', show_edges=True)
_ = pl.add_mesh(obj2, cmap='jet', show_edges=True)
_ = pl.add_mesh(obj3, cmap='jet', show_edges=True)
pl.add_bounding_box()
pl.show_grid()
pl.show()

import sys 
sys.exit() 

init, nn = stax.serial(stax.Dense(8), stax.Relu, stax.Dense(10), stax.Relu, stax.Dense(1))
pinn = pinns.FunctionSpaceNN((init, nn), ((-1,1), (-2,1), (0, 1)))
ws = pinn.init_weights(jax.random.PRNGKey(12234534))

x1, x2, x3 = np.meshgrid(np.linspace(-1,1,64), np.linspace(-2,1,63), np.linspace(0,1,65))
x_in = np.concatenate((x1.reshape([-1,1], order='F'), x2.reshape([-1,1], order='F'), x3.reshape([-1,1], order='F')), 1)

vals = pinn.neural_network(ws, x_in)

obj1 = pyvista.StructuredGrid(x1, x2, x3).cast_to_unstructured_grid()
obj1.point_data['value'] = vals.flatten(order='F')

facefun = pinn.interface_function((1,),(0,),(1,),(-1,),((-1,1),(-1,1),(0,1)))

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
