def transform(grid):
    import numpy as np
    grid = np.array(grid)
    out = grid.copy()
    
    # Find the row of 5's (fixed row)
    row5 = np.where(np.all(grid == 5, axis=1))[0][0]
    # Find the bottom row (last non-zero row in last column)
    bottom_row = np.max(np.where(grid[:, -1] != 0)[0])
    
    # The row above the bottom row contains the reference numbers
    ref_row = bottom_row - 1
    ref_values = grid[ref_row]
    
    # The top section is above the row of 5's
    top_section_height = row5
    # The bottom section is between row5+1 and ref_row-1
    bottom_section_start = row5 + 1
    bottom_section_end = ref_row - 1
    
    # Process each column
    for col in range(grid.shape[1]):
        ref_val = ref_values[col]
        if ref_val == 0:
            continue
            
        # Count 8's in top section of this column
        count_8 = np.sum(grid[:row5, col] == 8)
        
        if count_8 == 0:
            # Special case: if no 8's, place ref_val in bottom section
            if bottom_section_start <= bottom_section_end:
                out[bottom_section_end, col] = ref_val
        else:
            # Place ref_val in top section, starting from bottom of top section
            start_row = row5 - 1
            for _ in range(count_8):
                if start_row >= 0:
                    out[start_row, col] = ref_val
                    start_row -= 1
    
    return out.tolist()