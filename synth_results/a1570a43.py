import numpy as np

def transform(grid):
    g = np.array(grid)
    out = g.copy()
    h, w = g.shape
    
    # Find 3s (frame corners) and determine inner frame region
    threes = np.argwhere(g == 3)
    if len(threes) == 0:
        return out.tolist()
    
    three_rows = sorted(set(threes[:, 0].tolist()))
    three_cols = sorted(set(threes[:, 1].tolist()))
    # Inner frame: between the two rows and two cols of 3s (exclusive)
    inner_r1 = three_rows[0] + 1
    inner_r2 = three_rows[-1] - 1
    inner_c1 = three_cols[0] + 1
    inner_c2 = three_cols[-1] - 1
    
    # Find 2-cells
    twos = np.argwhere(g == 2)
    if len(twos) == 0:
        return out.tolist()
    
    two_min_r = int(twos[:, 0].min())
    two_min_c = int(twos[:, 1].min())
    two_max_r = int(twos[:, 0].max())
    two_max_c = int(twos[:, 1].max())
    
    # Compute shift needed to bring 2-shape inside frame
    dr = 0
    dc = 0
    if two_min_r < inner_r1:
        dr = inner_r1 - two_min_r
    elif two_max_r > inner_r2:
        dr = inner_r2 - two_max_r
    if two_min_c < inner_c1:
        dc = inner_c1 - two_min_c
    elif two_max_c > inner_c2:
        dc = inner_c2 - two_max_c
    
    # Clear original 2-cells
    for r, c in twos.tolist():
        out[r, c] = 0
    
    # Place 2-cells at new positions
    for r, c in twos.tolist():
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out[nr, nc] = 2
    
    return out.tolist()
