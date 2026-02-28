from collections import deque

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [list(row) for row in grid]
    bg = 7
    
    # Find the frame (rectangle of 6s)
    six_cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 6]
    if not six_cells:
        return result
    
    r1 = min(r for r,c in six_cells)
    r2 = max(r for r,c in six_cells)
    c1 = min(c for r,c in six_cells)
    c2 = max(c for r,c in six_cells)
    
    # Find the "fill" shape (non-6, non-bg cells)
    fill_cells = [(r,c,grid[r][c]) for r in range(rows) for c in range(cols) 
                  if grid[r][c] != bg and grid[r][c] != 6]
    
    if not fill_cells:
        return result
    
    fill_color = fill_cells[0][2]
    fill_rows = [r for r,c,v in fill_cells]
    fill_cols = [c for r,c,v in fill_cells]
    fill_h = max(fill_rows) - min(fill_rows) + 1
    fill_w = max(fill_cols) - min(fill_cols) + 1
    
    # Interior of the frame
    int_h = r2 - r1 - 1
    int_w = c2 - c1 - 1
    
    # Fill the interior with fill_color (scaled to fit)
    for r in range(r1+1, r2):
        for c in range(c1+1, c2):
            result[r][c] = fill_color
    
    # Remove the fill shape from its original position
    for r,c,v in fill_cells:
        result[r][c] = bg
    
    return result
