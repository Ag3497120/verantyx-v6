import numpy as np

def transform(grid):
    g = np.array(grid)
    out = g.copy()
    h, w = g.shape
    
    twos = np.argwhere(g == 2)  # 2-cells (obstacle/target)
    fives = np.argwhere(g == 5)  # 5-cells to move
    
    if len(twos) == 0 or len(fives) == 0:
        return out.tolist()
    
    def is_adjacent(r, c, twos):
        for tr, tc in twos:
            if abs(r - tr) <= 1 and abs(c - tc) <= 1:
                return True
        return False
    
    def nearest_two(r, c, twos):
        dists = [abs(r-tr) + abs(c-tc) for tr, tc in twos]
        idx = np.argmin(dists)
        return twos[idx]
    
    # Clear all 5s from output
    for r, c in fives:
        out[r, c] = 0
    
    # Move each 5 toward nearest 2-cell until adjacent
    for r, c in fives.tolist():
        r, c = int(r), int(c)
        tr, tc = [int(x) for x in nearest_two(r, c, twos.tolist())]
        
        # Direction toward nearest 2-cell
        dr = 0 if tr == r else (1 if tr > r else -1)
        dc = 0 if tc == c else (1 if tc > c else -1)
        
        # Move until adjacent to any 2-cell
        cr, cc = r, c
        while not is_adjacent(cr, cc, twos.tolist()):
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            cr, cc = nr, nc
        
        out[cr, cc] = 5
    
    return out.tolist()
