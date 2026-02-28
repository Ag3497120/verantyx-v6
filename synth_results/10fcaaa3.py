def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    
    # Tile the grid 2x2
    tiled = np.tile(g, (2, 2))
    out_rows, out_cols = tiled.shape
    result = tiled.copy()
    
    # For each non-zero cell in tiled, place 8 at diagonal neighbors (if currently 0)
    nz_positions = [(r, c) for r in range(out_rows) for c in range(out_cols) if tiled[r, c] != 0]
    
    for r, c in nz_positions:
        for dr, dc in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < out_rows and 0 <= nc < out_cols and result[nr, nc] == 0:
                result[nr, nc] = 8
    
    return result.tolist()
