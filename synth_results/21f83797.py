import numpy as np

def transform(grid):
    """Draw rectangle with cross-lines between two marked points."""
    grid = np.array(grid)
    h, w = grid.shape
    result = np.zeros((h, w), dtype=int)
    
    # Find the two marked points (value 2)
    points = np.argwhere(grid == 2)
    if len(points) < 2:
        return grid.tolist()
    
    (r1, c1), (r2, c2) = points[0], points[1]
    min_r, max_r = min(r1, r2), max(r1, r2)
    min_c, max_c = min(c1, c2), max(c1, c2)
    
    # Draw complete horizontal lines
    result[min_r, :] = 2
    result[max_r, :] = 2
    
    # Draw complete vertical lines  
    result[:, min_c] = 2
    result[:, max_c] = 2
    
    # Fill interior rectangle with 1s
    result[min_r+1:max_r, min_c+1:max_c] = 1
    
    # Ensure the cross lines remain 2 (in case they were overwritten)
    result[min_r, :] = 2
    result[max_r, :] = 2
    result[:, min_c] = 2
    result[:, max_c] = 2
    
    return result.tolist()
