import numpy as np

def transform(grid):
    h, w = len(grid), len(grid[0])
    result = [[7] * w for _ in range(h)]
    
    # Find vertical stripes
    for col in range(w):
        color = None
        start_row = None
        
        for row in range(h):
            current = grid[row][col]
            if current != 7:
                if color is None:
                    color = current
                    start_row = row
                elif current != color:
                    # Different color in same column - treat as separate stripe
                    if color is not None:
                        # Process previous stripe
                        new_col = (col + 2) % w
                        for r in range(start_row, row):
                            result[r][new_col] = color
                        color = current
                        start_row = row
            else:
                if color is not None:
                    # End of stripe
                    new_col = (col + 2) % w
                    for r in range(start_row, row):
                        result[r][new_col] = color
                    color = None
                    start_row = None
        
        # Handle stripe reaching bottom
        if color is not None:
            new_col = (col + 2) % w
            for r in range(start_row, h):
                result[r][new_col] = color
    
    return result