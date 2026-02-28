import numpy as np

def transform(grid):
    grid = np.array(grid, dtype=int)
    h, w = grid.shape
    
    # Mark obstacles: 2x2 blocks of 5s
    obstacle = np.zeros((h, w), dtype=bool)
    for y in range(h-1):
        for x in range(w-1):
            if (grid[y, x] == 5 and grid[y, x+1] == 5 and
                grid[y+1, x] == 5 and grid[y+1, x+1] == 5):
                obstacle[y:y+2, x:x+2] = True
    
    # Process each column
    for x in range(w):
        # Find single cells in this column (bottom to top processing is easier)
        for y in range(h-1, -1, -1):
            val = grid[y, x]
            if val != 0 and val != 5 and not obstacle[y, x]:
                # This is a single cell to move
                # Find highest empty spot above it
                new_y = y
                while new_y > 0:
                    if grid[new_y-1, x] != 0 or obstacle[new_y-1, x]:
                        break
                    new_y -= 1
                if new_y != y:
                    # Move it
                    grid[new_y, x] = val
                    grid[y, x] = 0
                # If new_y == y, it's already at top or blocked immediately above
    
    return grid.tolist()