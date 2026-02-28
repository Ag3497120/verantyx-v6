def transform(grid):
    import numpy as np
    grid = np.array(grid, dtype=int)
    h, w = grid.shape
    
    # Find all 4 positions
    positions = []
    for y in range(h):
        for x in range(w):
            if grid[y, x] == 4:
                positions.append((y, x))
    
    # Create output grid
    out_h = 2 * h + 3
    out_w = 2 * w + 3
    output = np.zeros((out_h, out_w), dtype=int)
    
    # For each 4 in input, place a plus pattern in output
    for y, x in positions:
        cy = 2 * y + 2  # center y in output
        cx = 2 * x + 2  # center x in output
        
        # Place plus pattern
        output[cy, cx] = 4
        if cy - 1 >= 0:
            output[cy - 1, cx] = 4
        if cy + 1 < out_h:
            output[cy + 1, cx] = 4
        if cx - 1 >= 0:
            output[cy, cx - 1] = 4
        if cx + 1 < out_w:
            output[cy, cx + 1] = 4
    
    return output.tolist()