import numpy as np
from collections import Counter

def transform(grid):
    arr = np.array(grid)
    rows, cols = arr.shape
    
    bg = Counter(arr.flatten().tolist()).most_common(1)[0][0]
    
    # Find frame height using 2-row pairs
    row_tuples = [tuple(arr[r]) for r in range(rows)]
    pair = (row_tuples[0], row_tuples[1])
    starts = [r for r in range(rows-1) if (row_tuples[r], row_tuples[r+1]) == pair]
    
    if len(starts) < 2:
        return grid
    
    frame_h = starts[1] - starts[0]
    frames = [arr[s:s+frame_h].copy() for s in starts if s+frame_h <= rows]
    
    if len(frames) < 3: return grid
    
    # Common vals (border markers) = in all frames
    def get_uvals(f):
        return set(f.flatten().tolist()) - {bg}
    common = set.intersection(*[get_uvals(f) for f in frames])
    
    def has_shapes(f):
        return any(f[r,c] not in common and f[r,c] != bg
                   for r in range(1, f.shape[0]-1) for c in range(f.shape[1]))
    
    non_empty = [(i,f) for i,f in enumerate(frames) if has_shapes(f)]
    if len(non_empty) < 2: return grid
    
    f1_i, f1 = non_empty[-2]
    f2_i, f2 = non_empty[-1]
    h, w = f1.shape
    
    result = f2.copy()
    shape_vals = (get_uvals(f2) | get_uvals(f1)) - common - {bg}
    
    for v in shape_vals:
        f1_cells = {(r,c) for r in range(h) for c in range(w) if f1[r,c] == v}
        f2_cells = {(r,c) for r in range(h) for c in range(w) if f2[r,c] == v}
        delta_cells = f2_cells - f1_cells
        if not delta_cells or not f1_cells: continue
        
        for (dr2, dc2) in delta_cells:
            dists = sorted((abs(dr2-r)+abs(dc2-c), r, c) for r,c in f1_cells)
            _, fr, fc = dists[0]
            nr, nc = dr2+(dr2-fr), dc2+(dc2-fc)
            if 0 <= nr < h and 0 <= nc < w:
                result[nr, nc] = v
    
    return result.tolist()
