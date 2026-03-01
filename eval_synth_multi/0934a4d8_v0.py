import numpy as np

def transform(grid_list):
    grid = np.array(grid_list)
    H, W = grid.shape
    eights = np.argwhere(grid == 8)
    r0, c0 = eights.min(axis=0)
    r1, c1 = eights.max(axis=0)
    
    # Detect symmetry axes from non-8 cells
    R = H + 1  # row axis
    C = W + 1  # col axis
    
    # Create a filled copy using all available symmetries
    filled = grid.copy()
    changed = True
    while changed:
        changed = False
        for r in range(r0, r1+1):
            for c in range(c0, c1+1):
                if filled[r][c] != 8:
                    continue
                # Try mirrors in priority order
                for mr, mc in [(r, C-c), (R-r, c), (R-r, C-c)]:
                    if 0 <= mr < H and 0 <= mc < W and filled[mr][mc] != 8:
                        filled[r][c] = filled[mr][mc]
                        changed = True
                        break
    
    # For remaining 8s, try transpose-like: g[r][c] = g[c][r] for border
    for r in range(r0, r1+1):
        for c in range(c0, c1+1):
            if filled[r][c] == 8:
                if c < 2 and r < H:
                    filled[r][c] = grid[c][r] if grid[c][r] != 8 else grid[c][R-r] if 0 <= R-r < H else 0
    
    result = filled[r0:r1+1, c0:c1+1].tolist()
    return [[int(v) for v in row] for row in result]
