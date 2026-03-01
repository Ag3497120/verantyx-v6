def transform(grid):
    rows, cols = len(grid), len(grid[0])
    bg = grid[0][0]
    
    # Find all non-bg colored shapes
    colors = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                colors.setdefault(grid[r][c], []).append((r, c))
    
    out = [row[:] for row in grid]
    
    for color, positions in colors.items():
        pos_set = set(positions)
        
        # Try all possible LR axes (integer and half-integer)
        min_c = min(c for _, c in positions)
        max_c = max(c for _, c in positions)
        
        best_axis = None
        best_kept = -1
        
        for axis_x2 in range(2 * min_c, 2 * max_c + 1):
            # axis = axis_x2 / 2
            kept = 0
            for r, c in positions:
                mirror_c = axis_x2 - c
                if (r, mirror_c) in pos_set:
                    kept += 1
            if kept > best_kept:
                best_kept = kept
                best_axis = axis_x2
        
        # Remove cells without mirror partner
        for r, c in positions:
            mirror_c = best_axis - c
            if (r, mirror_c) not in pos_set:
                out[r][c] = bg
    
    return out
