def transform(grid):
    import numpy as np
    from collections import Counter
    g = np.array(grid)
    rows, cols = g.shape
    
    # Find background color (most common)
    counts = Counter(g.flatten().tolist())
    bg = counts.most_common(1)[0][0]
    
    # Find separator rows: all cells != background
    sep_rows = [r for r in range(rows) if all(g[r, c] != bg for c in range(cols))]
    sep_cols = [c for c in range(cols) if all(g[r, c] != bg for r in range(rows))]
    
    # Number of bands in each dimension
    n_row_bands = len(sep_rows) + 1
    n_col_bands = len(sep_cols) + 1
    
    # Output: grid of (n_row_bands, n_col_bands) filled with background
    result = [[bg] * n_col_bands for _ in range(n_row_bands)]
    return result
