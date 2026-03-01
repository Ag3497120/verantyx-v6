def transform(grid):
    grid = [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    
    # For each column, move 2s upward to fill 0s
    for col in range(w):
        # Collect all values in this column
        column_vals = [grid[row][col] for row in range(h)]
        
        # Count 2s
        twos_count = column_vals.count(2)
        
        # Find the topmost 1 and bottommost 1
        ones_rows = [r for r in range(h) if grid[r][col] == 1]
        if not ones_rows:
            # No 1s in this column, clear all 2s
            for r in range(h):
                if grid[r][col] == 2:
                    grid[r][col] = 0
            continue
        
        top_one = min(ones_rows)
        bottom_one = max(ones_rows)
        
        # Find 0s within the 1-block
        zeros_in_block = []
        for r in range(top_one, bottom_one + 1):
            if grid[r][col] == 0:
                zeros_in_block.append(r)
        
        # Place 2s in these 0-positions
        for i, row in enumerate(zeros_in_block):
            if i < twos_count:
                grid[row][col] = 2
        
        # Clear all original 2s below the 1-block
        for r in range(h):
            if r > bottom_one and grid[r][col] == 2:
                grid[r][col] = 0
        
        # If we have more 2s than zeros in block, place remaining below bottom_one
        remaining_twos = twos_count - len(zeros_in_block)
        if remaining_twos > 0:
            placed = 0
            for r in range(bottom_one + 1, h):
                if grid[r][col] == 0 and placed < remaining_twos:
                    grid[r][col] = 2
                    placed += 1
    
    return grid
