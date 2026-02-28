def transform(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    rows, cols = g.shape
    bg = 8
    
    # Direction mapping: value -> (dr, dc, step)
    moves = {
        1: (0, 1, 1),   # right 1
        2: (0, -1, 2),  # left 2
        7: (-1, 0, 2),  # up 2
        9: (1, 0, 2),   # down 2
    }
    
    result = np.full_like(g, bg)
    
    for v, (dr, dc, step) in moves.items():
        cells = [(r,c) for r in range(rows) for c in range(cols) if g[r,c] == v]
        for r,c in cells:
            nr, nc = r + dr*step, c + dc*step
            if 0 <= nr < rows and 0 <= nc < cols:
                result[nr,nc] = v
    
    return result.tolist()
