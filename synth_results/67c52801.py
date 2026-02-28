import numpy as np
from scipy.ndimage import label

def transform(grid):
    g = np.array(grid)
    H, W = g.shape
    floor_color = g[H-1, 0]
    
    gap_row = None
    for r in range(H-2, -1, -1):
        if any(g[r,c] == floor_color for c in range(W)) and any(g[r,c] == 0 for c in range(W)):
            gap_row = r
            break
    if gap_row is None:
        return grid
    
    gaps = []
    c = 0
    while c < W:
        if g[gap_row, c] == 0:
            c_start = c
            while c < W and g[gap_row, c] == 0: c += 1
            gaps.append((c_start, c-1, c-c_start))
        else: c += 1
    
    obj_mask = (g != 0) & (g != floor_color)
    struct4 = np.array([[0,1,0],[1,1,1],[0,1,0]])
    labeled, n = label(obj_mask, structure=struct4)
    
    objects = []
    for lbl in range(1, n+1):
        mask = labeled == lbl
        rows, cols = np.where(mask)
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        sub = g[r0:r1+1, c0:c1+1].copy()
        objects.append(sub)
    
    out = np.where(obj_mask, 0, g)
    
    # Find matching: assign each object to a gap
    # Use backtracking
    assignment = {}  # obj_idx -> (gap_idx, rotated)
    
    def can_assign(obj_idx, gap_idx, sub):
        gw = gaps[gap_idx][2]
        if sub.shape[1] == gw:
            return (sub, False)
        rs = np.rot90(sub)
        if rs.shape[1] == gw:
            return (rs, True)
        return None
    
    def backtrack(obj_idx, used_gaps):
        if obj_idx == len(objects):
            return True
        sub = objects[obj_idx]
        for gi in range(len(gaps)):
            if gi in used_gaps:
                continue
            result = can_assign(obj_idx, gi, sub)
            if result:
                s, rot = result
                assignment[obj_idx] = (gi, s)
                used_gaps.add(gi)
                if backtrack(obj_idx + 1, used_gaps):
                    return True
                del assignment[obj_idx]
                used_gaps.remove(gi)
        return False
    
    backtrack(0, set())
    
    for obj_idx, (gi, sub) in assignment.items():
        gs = gaps[gi][0]
        oh, ow = sub.shape
        place_r_top = gap_row - oh + 1
        for dr in range(oh):
            for dc in range(ow):
                r = place_r_top + dr
                c = gs + dc
                if sub[dr, dc] != 0 and 0 <= r < H and 0 <= c < W:
                    out[r, c] = int(sub[dr, dc])
    
    return out.tolist()
