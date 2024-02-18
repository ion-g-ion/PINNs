from ._geometry import Patch, PatchNURBS, PatchBezier, PatchTensorProduct 
import numpy as np
import pickle

def save_patch(fname: str, patch: Patch):
    """
    Save a geometry patch in a file.
    The data is stored a a python dickt using pickle.

    Args:
        fname (str): the filename.
        patch (Patch): the patch to ba saved.
    """
    assert patch.dparam == 0, "Parameter dependent patches cannot be saved"
    
    if isinstance(patch, PatchNURBS):
        data = dict()
        data['type'] = 'PatchNURBS'
        data['bases'] = [('BSplineBasisJAX', b.deg, np.array(b.knots)) for b in patch.basis]
        data['weights'] = np.array(patch.weights())
        data['control_ponts'] = np.array(patch.control_points())
        data['dembedding'] = patch.dembedding
        
        with open(fname, 'wb') as file: 
            pickle.dump(data, file) 
    
def load_patch(fname) -> Patch:
    
    pass 
