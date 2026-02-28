def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    
    # Replace 1 with 2, extend by 3 more rows following the period
    # Find the row period
    ones = (g == 1).astype(int)
    
    # Find minimum period
    period = 1
    for p in range(1, rows+1):
        is_period = True
        for r in range(rows):
            if not np.array_equal(ones[r], ones[r % p]):
                is_period = False
                break
        if is_period:
            period = p
            break
    
    # Build output: original rows + 3 more rows following period
    result_rows = []
    for r in range(rows + 3):
        orig_row = list(g[r % period])
        result_rows.append([2 if v == 1 else v for v in orig_row])
    
    return result_rows
