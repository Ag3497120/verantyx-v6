import numpy as np

def transform(grid):
    """Draw cross-lines through colored pixels."""
    grid = np.array(grid)
    result = np.zeros_like(grid)
    h, w = grid.shape
    
    # Find the two colored pixels
    colors = {}
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0:
                colors[grid[r, c]] = (r, c)
    
    if len(colors) == 2:
        color_vals = list(colors.keys())
        c1, c2 = color_vals[0], color_vals[1]
        r1, col1 = colors[c1]
        r2, col2 = colors[c2]
        
        # Draw cross for color 1
        result[r1, :] = c1  # horizontal line
        result[:, col1] = c1  # vertical line
        
        # Draw cross for color 2
        result[r2, :] = c2  # horizontal line
        result[:, col2] = c2  # vertical line
        
        # At intersections, place color 2
        result[r1, col2] = 2  # color1's horizontal crosses color2's vertical
        result[r2, col1] = 2  # color2's horizontal crosses color1's vertical
    
    return result.tolist()
