import numpy as np

def transform(grid):
    grid = np.array(grid, dtype=int)
    h, w = grid.shape
    center = h // 2  # 7 in 0-index for 15x15? Wait, 15//2 = 7, but cross is at row 3? Let's check.
    # Actually, looking at examples: cross is at row 3 (0-index) and col 3.
    # Because 5s are at row 3 all columns, and col 3 all rows.
    cross_row = 3
    cross_col = 3
    
    # Mark cross cells
    cross_cells = set()
    for r in range(h):
        cross_cells.add((r, cross_col))
    for c in range(w):
        cross_cells.add((cross_row, c))
    
    # For each cell not in cross, check if it's part of a horizontal or vertical group of same value length >= 2
    output = grid.copy()
    for r in range(h):
        for c in range(w):
            if (r, c) in cross_cells:
                continue
            val = grid[r, c]
            if val == 0:
                continue
            
            # Check horizontal group length
            left = c
            while left - 1 >= 0 and grid[r, left - 1] == val:
                left -= 1
            right = c
            while right + 1 < w and grid[r, right + 1] == val:
                right += 1
            h_len = right - left + 1
            
            # Check vertical group length
            up = r
            while up - 1 >= 0 and grid[up - 1, c] == val:
                up -= 1
            down = r
            while down + 1 < h and grid[down + 1, c] == val:
                down += 1
            v_len = down - up + 1
            
            # If both lengths are 1, it's isolated
            if h_len == 1 and v_len == 1:
                output[r, c] = 0
    return output.tolist()