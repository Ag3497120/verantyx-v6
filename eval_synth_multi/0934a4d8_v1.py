import numpy as np

def transform(grid_list):
    grid = np.array(grid_list)
    H, W = grid.shape
    eights = np.argwhere(grid == 8)
    r0, c0 = eights.min(axis=0)
    r1, c1 = eights.max(axis=0)
    R, C = H + 1, W + 1
    
    filled = grid.copy()
    # Multi-pass fill using symmetry
    for _ in range(3):
        for r in range(H):
            for c in range(W):
                if filled[r][c] != 8:
                    continue
                for mr, mc in [(R-r, c), (r, C-c), (R-r, C-c), (c, r), (c, C-r)]:
                    if 0 <= mr < H and 0 <= mc < W and filled[mr][mc] != 8:
                        filled[r][c] = filled[mr][mc]
                        break
    
    return [[int(v) for v in row] for row in filled[r0:r1+1, c0:c1+1]]
