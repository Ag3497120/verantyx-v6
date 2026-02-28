def transform(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Find lookup table: rows where col 0 != 0
    lookup_seq = []
    for r in range(rows):
        if grid[r][0] != 0:
            lookup_seq.append((grid[r][0], grid[r][1]))
    
    # Find divider: column that is all same non-zero value
    divider = None
    for c in range(cols):
        col_vals = [grid[r][c] for r in range(rows)]
        if len(set(col_vals)) == 1 and col_vals[0] != 0:
            divider = c
            break
    
    if divider is None:
        # Try: divider is second non-trivial column
        divider = 9  # default
    
    result = [row[:] for row in grid]
    
    for r in range(rows):
        for c in range(divider+1, cols):
            v = grid[r][c]
            if v != 0:
                # Apply sequential lookup
                cur = v
                for a, b in lookup_seq:
                    if cur == a:
                        cur = b
                result[r][c] = cur
    return result
