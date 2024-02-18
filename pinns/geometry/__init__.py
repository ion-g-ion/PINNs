"""
Geometry module
"""
from ._geometry import PatchNURBS, PatchBezier, Patch, PatchTensorProduct, IsogeometricForm
from ._multipatch import PatchConnectivity, check_match, match_patches
from ._projections import gap_to_convex_polytope
from ._factory import box_patch
from ._loader_saver import save_patch, load_patch

__all__ = ['PatchNURBS', 'PatchConnectivity', 'PatchBezier', 'Patch', 'IsogeometricForm', 'check_match', 'match_patches', 'gap_to_convex_polytope', 'box_patch', 'PatchTensorProduct']
__all__ += ['save_patch', 'load_patch']