import numpy as np

def transform(grid):
    """Connect aligned pairs of 8s with 3s."""
    grid = np.array(grid)
    result = grid.copy()
    h, w = grid.shape
    
    # Find all 8s
    positions = list(zip(*np.where(grid == 8)))
    
    # For each pair of 8s
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            r1, c1 = positions[i]
            r2, c2 = positions[j]
            
            # If same row, connect horizontally
            if r1 == r2:
                min_c, max_c = min(c1, c2), max(c1, c2)
                for c in range(min_c+1, max_c):
                    result[r1, c] = 3
            
            # If same column, connect vertically
            elif c1 == c2:
                min_r, max_r = min(r1, r2), max(r1, r2)
                for r in range(min_r+1, max_r):
                    result[r, c1] = 3
    
    return result.tolist()
