from typing import Tuple, TypedDict, Sequence, Dict
from ._geometry import PatchNURBS, PatchNURBSParam, Patch
import numpy as np
import itertools 

# Patch connectivity object
PatchConnectivity = TypedDict('PatchConnectivity', {'first': str, 'second': str, 'axis_first': Tuple[int], 'axis_second': Tuple[int], 'end_first': Tuple[int], 'end_second': Tuple[int], 'axis_permutation': Tuple[Tuple[int,int]]})

def match_patches(patches: Dict[str, Patch], eps: float=1e-7, verbose: bool=False) -> Sequence[PatchConnectivity]:
    """
    Find the connectivitty between the given patches.

    Args:
        patches (Dict[str, Patch]): dictionary og geometry patches.
        eps (float, optional): tolerance for finding the match between vertices in the physical domain. Defaults to 1e-7.
        verbose (bool, optional): show extra info. Defaults to False.

    Returns:
        Sequence[PatchConnectivity]: the connectivity info.
    """
    names = list(patches.keys())
    
    conns = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            conn = check_match(names[i], patches[names[i]], names[j], patches[names[j]], eps, verbose)
            if conn is not None:
                conns.append(conn)
    
    return conns

def check_match(name1: str, patch1: PatchNURBSParam, name2: str, patch2: PatchNURBSParam, eps: float=1e-7, verbose: bool=True) -> PatchConnectivity | None:
    
    assert patch1.d == patch2.d, "Patches must ave same dimensionality"
    assert patch1.dembedding == patch2.dembedding, "The patches must live in the same space"
    d = patch1.d 
    de = patch1.dembedding
    assert d==2 or d==3, "Manifold must be 3d or 2d."
    
    if verbose: print('Checking %s against %s'%(str(name1), str(name2)))
    
    # list of all vertices
    vertices1 = []
    vertices2 = [] 
    
    lst = [[0,1]]*d
    for bin in itertools.product(*lst):
        pt = patch1(np.array([patch1.domain[k][bin[k]] for k in range(3)]).reshape([1,-1]))
        vertices1.append({'point': np.array(pt).flatten(), 'ends': bin, 'point_reference': np.array([patch1.domain[k][bin[k]] for k in range(d)]).reshape([1,-1])})
    
        pt = patch2(np.array([patch2.domain[k][bin[k]] for k in range(3)]).reshape([1,-1]))
        vertices2.append({'point': np.array(pt).flatten(), 'ends': bin, 'point_reference': np.array([patch2.domain[k][bin[k]] for k in range(d)]).reshape([1,-1])})

    common = []
    for v1 in vertices1:
        for v2 in vertices2:
            #if verbose: print("Vertex1 %s, vertex2 %s"%(str(v1), str(v2)))
            if np.linalg.norm(v1['point']-v2['point']) < eps:
                #if verbose: print("\tare common")
                common.append((v1, v2))
    
    if verbose: print("\nFound %d vertices in common"%(len(common)))

    if len(common) == 0:
        return None 
    
    middle1 = 0
    middle2 = 0
    for c in common:
        middle1 += c[0]['point_reference']
        middle2 += c[1]['point_reference']
    middle1 /= len(common)
    middle2 /= len(common)
    
    middle1 = common[0][0]['point_reference']
    middle2 = common[0][1]['point_reference']

    coordinate1 = patch1.GetJacobian(middle1).reshape([de, d])
    coordinate2 = patch2.GetJacobian(middle2).reshape([de, d]) 
    # normalize 
    coordinate1 = coordinate1.T / np.linalg.norm(coordinate1, axis=1)
    coordinate2 = coordinate2.T / np.linalg.norm(coordinate2, axis=1)
    
    coordinate2 = coordinate1.T @ coordinate2
    
    if verbose: print('Coordinate system 1:', coordinate1)
    if verbose: print('Coordinate system 2:', coordinate2)
    perm =  []
    for i in range(d):
        a2 = int(np.where(np.abs(coordinate2[i,:])>0.1)[0])
        direction = -1 if coordinate2[i,a2] < 0 else 1
        perm.append((a2, direction))
    perm = tuple(perm)
    
    conn_data = {'first': name1, 'second': name2, 'axis_permutation': perm}
    
    lst1 = []
    lst2 = []

    for c in common:
        lst1.append(c[0]['ends'])
        lst2.append(c[1]['ends'])
        
    bool1 = np.all(np.array(lst1) == np.array(lst1)[0,:], axis=0)
    bool2 = np.all(np.array(lst2) == np.array(lst2)[0,:], axis=0)
    
    
    conn_data['axis_first'] = tuple(int(i) for i in np.where(bool1)[0])
    conn_data['axis_second'] = tuple(int(i) for i in np.where(bool2)[0])
    
    conn_data['end_first'] = tuple(int(-i) for i in np.array(lst1)[0,conn_data['axis_first']])
    conn_data['end_second'] = tuple(int(-i) for i in np.array(lst2)[0,conn_data['axis_second']])
    
    if verbose:
        print('Axis first :', conn_data['axis_first'])
        print('End first  :', conn_data['end_first'])
        print('Axis second:', conn_data['axis_second'])
        print('End second :', conn_data['end_second'])
        for i in range(d):
            print('First patch axis %d -> second patch axis %d, direction %s'%(i, conn_data['axis_permutation'][i][0], "SAME" if conn_data['axis_permutation'][i][1]==1 else "REVERSED"))

    return conn_data