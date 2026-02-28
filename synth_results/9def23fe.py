def transform(grid):
    from collections import Counter
    rows = len(grid)
    cols = len(grid[0])
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find the main block (most common non-bg value)
    non_bg = [v for v in flat if v != bg]
    block_val = Counter(non_bg).most_common(1)[0][0]
    blocker_val = None  # any other non-bg value
    for v in set(non_bg):
        if v != block_val:
            blocker_val = v
            break
    
    block_cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == block_val]
    br_min = min(r for r,c in block_cells)
    br_max = max(r for r,c in block_cells)
    bc_min = min(c for r,c in block_cells)
    bc_max = max(c for r,c in block_cells)
    
    result = [row[:] for row in grid]
    
    def is_blocker(v):
        return v == blocker_val
    
    # For each row of the block: extend LEFT/RIGHT
    for r in range(br_min, br_max+1):
        left_has_blocker = any(is_blocker(grid[r][c]) for c in range(0, bc_min))
        right_has_blocker = any(is_blocker(grid[r][c]) for c in range(bc_max+1, cols))
        if not left_has_blocker:
            for c in range(0, bc_min):
                if result[r][c] == bg:
                    result[r][c] = block_val
        if not right_has_blocker:
            for c in range(bc_max+1, cols):
                if result[r][c] == bg:
                    result[r][c] = block_val
    
    # For each col of the block: extend UP/DOWN
    for c in range(bc_min, bc_max+1):
        above_has_blocker = any(is_blocker(grid[r][c]) for r in range(0, br_min))
        below_has_blocker = any(is_blocker(grid[r][c]) for r in range(br_max+1, rows))
        if not above_has_blocker:
            for r in range(0, br_min):
                if result[r][c] == bg:
                    result[r][c] = block_val
        if not below_has_blocker:
            for r in range(br_max+1, rows):
                if result[r][c] == bg:
                    result[r][c] = block_val
    
    return result
