import numpy as np

def transform(grid):
    """Mark rows where all values are identical."""
    grid = np.array(grid)
    result = []
    
    for row in grid:
        if len(set(row)) == 1:  # All values same
            result.append([5, 5, 5])
        else:
            result.append([0, 0, 0])
    
    return result
