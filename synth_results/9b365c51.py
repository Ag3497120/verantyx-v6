import numpy as np
from collections import Counter, defaultdict

def transform(grid):
    g = np.array(grid)
    out = g.copy()
    h, w = g.shape
    
    # Find key columns: columns where all non-zero values are the same (non-8)
    key_cols = []  # [(col, color), ...]
    for c in range(w):
        col_vals = [v for v in g[:, c].tolist() if v != 0]
        if col_vals and all(v != 8 for v in col_vals) and len(set(col_vals)) == 1:
            key_cols.append((c, col_vals[0]))
    
    # Sort keys by column position
    key_cols.sort()
    key_colors = [color for _, color in key_cols]
    key_col_positions = set(col for col, _ in key_cols)
    
    # Clear key columns in output
    for col in key_col_positions:
        out[:, col] = 0
    
    # Find rectangular blocks of 8s by grouping columns by their row-pattern
    # col -> frozenset of rows with 8
    col_to_rows = {}
    for c in range(w):
        rows_with_8 = frozenset(r for r in range(h) if g[r, c] == 8)
        if rows_with_8:
            col_to_rows[c] = rows_with_8
    
    if not col_to_rows:
        return out.tolist()
    
    # Group columns that share the same row pattern
    row_pattern_to_cols = defaultdict(list)
    for c, rows in col_to_rows.items():
        row_pattern_to_cols[rows].append(c)
    
    # Each group forms a rectangle, split by contiguous column ranges
    rectangles = []
    for rows_set, cols in row_pattern_to_cols.items():
        cols.sort()
        # Split into contiguous groups
        groups = []
        start = cols[0]
        prev = cols[0]
        for c in cols[1:]:
            if c != prev + 1:
                groups.append(list(range(start, prev+1)))
                start = c
            prev = c
        groups.append(list(range(start, prev+1)))
        
        for g_cols in groups:
            rectangles.append((g_cols[0], rows_set, g_cols))
    
    # Sort rectangles by their leftmost column
    rectangles.sort(key=lambda x: x[0])
    
    # Assign key colors to rectangles by order
    for i, (min_col, rows_set, cols) in enumerate(rectangles):
        if i < len(key_colors):
            color = key_colors[i]
        else:
            break
        for r in rows_set:
            for c in cols:
                out[r, c] = color
    
    return out.tolist()
