import pyvista as pv 
import numpy as np
from ..geometry import Patch
from ..pinns import FunctionSpaceNN
from typing import Sequence, Dict, Callable

def plot(geom_patch: Patch, functions: Dict[str, Callable]=dict(), N: Sequence[int]|int=32) -> pv.StructuredGrid:
    """
    Plot a pyvista object from a geometry patch. 
    The parameter dependence should not appear.

    Args:
        geom_patch (PatchNURBS | PatchNURBSParam | Callable): the 
        functions (Dict[str, Callable], optional): functions that can be attached to the object. Defaults to dict().
        N (Sequence[int] | int, optional): the discretization. If int is provided an uniform meshgrid is created. Defaults to 32.

    Returns:
        pv.StructuredGrid: _description_
    """
    if isinstance(N, int):
        N = geom_patch.d*[N]
    
    ys = [np.linspace(float(geom_patch.domain[i][0]), float(geom_patch.domain[i][1]), N[i]) for i in range(geom_patch.d)]
    xs = np.meshgrid(*ys)
    x_in = np.concatenate(tuple(xs[i].reshape([-1,1], order='F') for i in range(geom_patch.d)), 1)
    
    xgs = geom_patch(x_in)
    x1 = np.array(xgs[:,0]).reshape(N, order='F')
    x2 = np.array(xgs[:,1]).reshape(N, order='F')
    x3 = x2*0 if geom_patch.d != 3 else np.array(xgs[:,2]).reshape(N, order='F')
    obj = pv.StructuredGrid(x1, x2, x3).cast_to_unstructured_grid()
    for key in functions:
        vals = functions[key](x_in).squeeze()
        obj.point_data[key] = vals
    
    return obj 