import numpy as np

def transform(grid):
    """Shift all rows down by 1, fill top with zeros."""
    grid = np.array(grid)
    h, w = grid.shape
    
    result = np.zeros((h, w), dtype=int)
    result[1:] = grid[:-1]
    
    return result.tolist()
