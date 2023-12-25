"""
Geometry module
"""
from ._geometry import PatchNURBS, PatchNURBSParam, AffineTransformation, Patch
from ._multipatch import PatchConnectivity, check_match, match_patches

__all__ = ['PatchNURBS', 'PatchNURBSParam', 'AffineTransformation', 'PatchConnectivity', 'Patch', 'check_match', 'match_patches']