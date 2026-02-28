def transform(grid):
    return _solve(grid)

def solve_770cc55f(grid):
    g = [list(r) for r in grid]
    H, W = len(g), len(g[0])
    out = [row[:] for row in g]
    
    # Find divider row (all 2s)
    div_row = next(r for r in range(H) if all(g[r][c]==2 for c in range(W)))
    
    # Find pattern rows above and below
    # Pattern = non-zero non-2 columns
    top_pattern_row = None
    bot_pattern_row = None
    for r in range(div_row):
        if any(g[r][c] not in [0,2] for c in range(W)):
            top_pattern_row = r; break
    for r in range(H-1, div_row, -1):
        if any(g[r][c] not in [0,2] for c in range(W)):
            bot_pattern_row = r; break
    
    if top_pattern_row is None or bot_pattern_row is None: return grid
    
    top_cols = {c for c in range(W) if g[top_pattern_row][c] not in [0,2]}
    bot_cols = {c for c in range(W) if g[bot_pattern_row][c] not in [0,2]}
    
    # Intersection
    intersection = top_cols & bot_cols
    
    # Larger pattern determines which side gets filled
    if len(top_cols) >= len(bot_cols):
        # Fill between top_pattern_row and divider
        fill_rows = range(top_pattern_row+1, div_row)
    else:
        # Fill between divider and bot_pattern_row
        fill_rows = range(div_row+1, bot_pattern_row)
    
    for r in fill_rows:
        for c in intersection:
            out[r][c] = 4
    
    return out


_solve = solve_770cc55f
