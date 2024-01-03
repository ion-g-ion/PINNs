"""
Geometry module
"""
from ._geometry import PatchNURBS, PatchNURBSParam, AffineTransformation, Patch
from ._multipatch import PatchConnectivity, check_match, match_patches
from ._projections import gap_to_convex_polytope
from ._factory import box_patch

__all__ = ['PatchNURBS', 'PatchNURBSParam', 'AffineTransformation', 'PatchConnectivity', 'Patch', 'check_match', 'match_patches', 'gap_to_convex_polytope', 'box_patch']