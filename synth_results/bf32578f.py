def transform(grid):
    g = [list(row) for row in grid]
    rows, cols = len(g), len(g[0])
    
    # Find bounding box of non-zero cells
    cells = [(r,c) for r in range(rows) for c in range(cols) if g[r][c] != 0]
    if not cells:
        return grid
    
    min_r = min(r for r,c in cells)
    max_r = max(r for r,c in cells)
    min_c = min(c for r,c in cells)
    max_c = max(c for r,c in cells)
    color = g[cells[0][0]][cells[0][1]]
    
    # Clear original shape
    for r in range(rows):
        for c in range(cols):
            g[r][c] = 0
    
    # Fill bounding box interior
    for r in range(min_r+1, max_r):
        for c in range(min_c+1, max_c):
            g[r][c] = color
    
    return g
