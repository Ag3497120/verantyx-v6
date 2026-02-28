import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid, dtype=int)
    h, w = grid.shape
    output = np.zeros_like(grid)
    
    # Find the bounding box of the 5's region
    mask_5 = (grid == 5)
    if not mask_5.any():
        return grid.tolist()
    
    rows, cols = np.where(mask_5)
    top, bottom = rows.min(), rows.max()
    left, right = cols.min(), cols.max()
    
    # Process each special color pixel on the boundary
    for y in range(h):
        for x in range(w):
            val = grid[y, x]
            if val != 0 and val != 5:
                # Check if pixel is adjacent to the 5's rectangle
                is_on_boundary = False
                direction = None
                
                # Left side
                if x == left - 1 and top <= y <= bottom:
                    is_on_boundary = True
                    direction = 'right'
                # Right side
                elif x == right + 1 and top <= y <= bottom:
                    is_on_boundary = True
                    direction = 'left'
                # Top side
                elif y == top - 1 and left <= x <= right:
                    is_on_boundary = True
                    direction = 'down'
                # Bottom side
                elif y == bottom + 1 and left <= x <= right:
                    is_on_boundary = True
                    direction = 'up'
                
                if is_on_boundary:
                    # Propagate the color
                    if direction == 'right':
                        for dx in range(0, right - left + 1):
                            if grid[y, left + dx] == 5:
                                output[y, left + dx] = val
                    elif direction == 'left':
                        for dx in range(0, right - left + 1):
                            if grid[y, right - dx] == 5:
                                output[y, right - dx] = val
                    elif direction == 'down':
                        for dy in range(0, bottom - top + 1):
                            if grid[top + dy, x] == 5:
                                output[top + dy, x] = val
                    elif direction == 'up':
                        for dy in range(0, bottom - top + 1):
                            if grid[bottom - dy, x] == 5:
                                output[bottom - dy, x] = val
    
    # Fill in any remaining 5's that weren't colored (shouldn't happen in these examples)
    output[mask_5 & (output == 0)] = 5
    
    return output.tolist()