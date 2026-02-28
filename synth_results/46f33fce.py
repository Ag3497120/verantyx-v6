def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Each non-zero cell at (r,c) [r,c odd] maps to a 4x4 block in the output
    # Output size = input_size * 2 (20x20 for 10x10 input)
    out_rows = rows * 2
    out_cols = cols * 2
    result = [[0]*out_cols for _ in range(out_rows)]
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                v = grid[r][c]
                # Map to super-cell
                sr = r // 2
                sc = c // 2
                # Fill 4x4 block
                for dr in range(4):
                    for dc in range(4):
                        rr = sr * 4 + dr
                        cc = sc * 4 + dc
                        if rr < out_rows and cc < out_cols:
                            result[rr][cc] = v
    
    return result
