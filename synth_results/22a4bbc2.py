def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    out = grid.copy()
    
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 1:
                # Check if this 1 is part of a horizontal block of 1s
                # that is directly above a horizontal block of 8s of same width
                # First find the horizontal extent of 1s starting at (r,c)
                if c > 0 and grid[r, c-1] == 1:
                    continue  # already processed as part of a block
                start_c = c
                end_c = c
                while end_c < w and grid[r, end_c] == 1:
                    end_c += 1
                block_len = end_c - start_c
                
                # Check if row r+1 exists and has block of 8s of same length at same columns
                if r+1 < h:
                    match = True
                    for k in range(block_len):
                        if grid[r+1, start_c + k] != 8:
                            match = False
                            break
                    if match:
                        # Also check that the block of 8s is exactly that length
                        # (i.e., not part of a longer block)
                        if start_c > 0 and grid[r+1, start_c-1] == 8:
                            match = False
                        if end_c < w and grid[r+1, end_c] == 8:
                            match = False
                        if match:
                            out[r, start_c:end_c] = 2
    return out.tolist()