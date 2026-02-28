import numpy as np

def transform(grid):
    grid = np.array(grid, dtype=int)
    h, w = grid.shape
    
    # Find bounding box of non-zero cells
    non_zero = np.where(grid != 0)
    if len(non_zero[0]) == 0:
        return grid.tolist()
    
    top = non_zero[0].min()
    bottom = non_zero[0].max()
    left = non_zero[1].min()
    right = non_zero[1].max()
    
    # Create output grid
    output = np.zeros_like(grid)
    
    # Copy the shape with alternating shifts
    for r in range(top, bottom + 1):
        row_data = grid[r, left:right+1].copy()
        
        # Top and bottom border rows stay unchanged
        if r == top or r == bottom:
            output[r, left:right+1] = row_data
            continue
            
        # Calculate row index within shape (0-based from top)
        rel_row = r - top
        
        # Determine shift direction
        if rel_row % 2 == 0:
            # Even rows within shape: no shift
            output[r, left:right+1] = row_data
        else:
            # Odd rows within shape: alternate left/right
            if (rel_row // 2) % 2 == 0:
                # Shift left
                output[r, left:right] = row_data[1:]
                output[r, right] = 0
            else:
                # Shift right
                output[r, left+1:right+1] = row_data[:-1]
                output[r, left] = 0
    
    return output.tolist()