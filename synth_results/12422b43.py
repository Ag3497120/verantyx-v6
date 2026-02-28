def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    result = g.copy()
    
    # Find the 5-column (column with 5s at the top)
    five_col = None
    for c in range(cols):
        if g[0, c] == 5:
            five_col = c
            break
    
    if five_col is None:
        return grid
    
    # Count height of 5s (N = period length)
    N = 0
    for r in range(rows):
        if g[r, five_col] == 5:
            N += 1
        else:
            break
    
    # The first N rows form the period (excluding the 5-col values)
    period = g[:N, :].copy()
    period[:, five_col] = 0  # don't include the 5-column in pattern
    
    # Find last non-zero row of the initial pattern (in non-5 cols)
    last_pattern_row = -1
    for r in range(rows):
        row_vals = [g[r, c] for c in range(cols) if c != five_col]
        if any(v != 0 for v in row_vals):
            last_pattern_row = r
    
    # Fill rows after last_pattern_row with repeating period
    for r in range(last_pattern_row + 1, rows):
        period_row = period[(r - (last_pattern_row + 1)) % N]
        for c in range(cols):
            if c != five_col and period_row[c] != 0:
                result[r, c] = period_row[c]
    
    return result.tolist()
