import numpy as np

def transform(grid_list):
    grid = np.array(grid_list)
    H, W = grid.shape
    eights = np.argwhere(grid == 8)
    r0, c0 = eights.min(axis=0)
    r1, c1 = eights.max(axis=0)
    R, C = H + 1, W + 1
    
    filled = grid.copy()
    for _ in range(5):
        for r in range(H):
            for c in range(W):
                if filled[r][c] != 8:
                    continue
                mirrors = [(R-r,c),(r,C-c),(R-r,C-c)]
                if c < 2:
                    mirrors.extend([(c,r),(c,R-r)])
                if r < 2:
                    mirrors.extend([(c,r),(C-c,r)])
                for mr, mc in mirrors:
                    if 0 <= mr < H and 0 <= mc < W and filled[mr][mc] != 8:
                        filled[r][c] = filled[mr][mc]
                        break
    return [[int(v) for v in row] for row in filled[r0:r1+1, c0:c1+1]]
