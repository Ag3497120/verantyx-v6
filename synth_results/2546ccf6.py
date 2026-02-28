import numpy as np

def transform(grid):
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find the dividing color (appears in full rows and columns)
    color_counts = {}
    for color in np.unique(grid):
        color_counts[color] = np.sum(grid == color)
    
    # The dividing color is the one that appears in complete rows and columns
    # It will typically be very frequent
    dividing_color = max(color_counts, key=color_counts.get)
    
    # Find rows and columns that are entirely the dividing color
    dividing_rows = [i for i in range(h) if all(grid[i, j] == dividing_color for j in range(w))]
    dividing_cols = [j for j in range(w) if all(grid[i, j] == dividing_color for i in range(h))]
    
    # Create output grid, starting with the original
    output = grid.copy()
    
    # Process each block between dividing lines
    for row_start, row_end in zip(dividing_rows[:-1], dividing_rows[1:]):
        for col_start, col_end in zip(dividing_cols[:-1], dividing_cols[1:]):
            # Extract the block (excluding the dividing lines themselves)
            block = grid[row_start+1:row_end, col_start+1:col_end]
            if block.size == 0:
                continue
            
            # Rotate the block 180 degrees
            rotated_block = np.rot90(block, 2)
            
            # Place it back in the output
            output[row_start+1:row_end, col_start+1:col_end] = rotated_block
    
    return output.tolist()