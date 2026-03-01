import numpy as np
from collections import defaultdict

def get_shape(cells):
    if not cells: return frozenset()
    min_r = min(r for r,c in cells)
    min_c = min(c for r,c in cells)
    return frozenset((r-min_r, c-min_c) for r,c in cells)

def find_components(grid, val):
    arr = np.array(grid)
    rows, cols = arr.shape
    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if arr[r,c] == val and (r,c) not in visited:
                comp = set()
                stack = [(r,c)]
                while stack:
                    rr,cc = stack.pop()
                    if (rr,cc) in visited or rr<0 or rr>=rows or cc<0 or cc>=cols: continue
                    if arr[rr,cc] != val: continue
                    visited.add((rr,cc)); comp.add((rr,cc))
                    for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]: stack.append((rr+dr,cc+dc))
                if comp: components.append(comp)
    return components

def transform(grid):
    arr = np.array(grid)
    rows, cols = arr.shape
    
    # Find all unique non-0, non-1 values
    vals = set(arr.flatten().tolist()) - {0, 1}
    
    # Build prototype shapes for each non-1, non-0 value
    proto_shapes = {}
    for v in vals:
        comps = find_components(grid, v)
        for comp in comps:
            s = get_shape(comp)
            proto_shapes[s] = v
    
    # Find all 1-components and match to prototypes
    result = arr.copy()
    one_comps = find_components(grid, 1)
    
    for comp in one_comps:
        s = get_shape(comp)
        if s in proto_shapes:
            color = proto_shapes[s]
            for r,c in comp:
                result[r,c] = color
    
    return result.tolist()
