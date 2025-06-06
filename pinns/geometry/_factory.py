import jax.numpy as jnp
import numpy as np
from ._geometry import PatchNURBS
from ..functions import BSplineBasisJAX
from typing import Tuple, Any
from ._base import rotation_matrix_3d




def box_patch(size: Tuple[float], rotation: Tuple[float] | None = None, position: Tuple[float] | None = None, dtype: np.dtype = np.float64) -> PatchNURBS:
    """
    Crete a NURBS box around the origin, i.e. the domain `[-size[0]/2, size[0]/2] x [-size[1]/2, size[1]/2] ... `.
    The domain can be rotated and translated.
    The Bspline basis used for every dimension is linear with 2 basis elements.

    Args:
        size (Tuple[float]): the sizes of the domain.
        rand_key (Any): random key.
        rotation (Tuple[float] | None, optional): rotation angles along the axes. Defaults to None.
        position (Tuple[float] | None, optional): translation. Defaults to None.
        dtype (np., optional): datatype. Defaults to np.flaot64.

    Returns:
        PatchNURBSParam: resulting patch.
    """
    basis = BSplineBasisJAX(np.array([-1, 1]), 1)
    d = len(size)

    control_pts = np.zeros([2]*d+[d])
    for i in range(d):
        idx = [slice(None)]*d
        idx[i] = 1
        control_pts[tuple(idx+[i])] = float(size[i])
    control_pts -= np.array(size).astype(dtype)/2

    if rotation is not None:
        if d==2:
            Rot = np.array([[np.cos(rotation[0]), -np.sin(rotation[0])],[np.sin(rotation[0]), np.cos(rotation[0])]])
        elif d==3:
            Rot = rotation_matrix_3d(rotation)
        control_pts = np.einsum('...n,mn->...m', control_pts, Rot)
    if position is not None:
        control_pts += np.array(position, dtype=dtype)
    weights = np.ones([2]*d)

    patch = PatchNURBS([basis]*d, control_pts, weights, 0, d)

    return patch
