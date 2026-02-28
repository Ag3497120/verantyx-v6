def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    result = g.copy()
    
    # Find "line" rows: rows where all non-zero cells have the same color
    h_lines = {}  # color -> set of rows
    for r in range(rows):
        nz_vals = set(int(g[r, c]) for c in range(cols) if g[r, c] != 0)
        if len(nz_vals) == 1:
            col = nz_vals.pop()
            h_lines.setdefault(col, []).append(r)
    
    # Find "line" cols: cols where all non-zero cells have the same color
    v_lines = {}  # color -> set of cols
    for c in range(cols):
        nz_vals = set(int(g[r, c]) for r in range(rows) if g[r, c] != 0)
        if len(nz_vals) == 1:
            col_val = nz_vals.pop()
            v_lines.setdefault(col_val, []).append(c)
    
    # Find isolated cells (not on a line row or line col)
    line_rows = set(r for rs in h_lines.values() for r in rs)
    line_cols = set(c for cs in v_lines.values() for c in cs)
    
    for r in range(rows):
        for c in range(cols):
            v = int(g[r, c])
            if v == 0:
                continue
            if r in line_rows or c in line_cols:
                continue  # on a line, keep as-is
            
            # Isolated cell - find matching line
            moved = False
            
            # Check horizontal lines for same color
            if v in h_lines:
                nearest_row = min(h_lines[v], key=lambda lr: abs(lr - r))
                # Move to adjacent to nearest_row, same column
                target_r = nearest_row - 1 if r < nearest_row else nearest_row + 1
                if 0 <= target_r < rows:
                    result[r, c] = 0
                    result[target_r, c] = v
                    moved = True
            
            # Check vertical lines for same color
            if not moved and v in v_lines:
                nearest_col = min(v_lines[v], key=lambda lc: abs(lc - c))
                target_c = nearest_col - 1 if c < nearest_col else nearest_col + 1
                if 0 <= target_c < cols:
                    result[r, c] = 0
                    result[r, target_c] = v
                    moved = True
            
            # No matching line -> remove cell
            if not moved:
                result[r, c] = 0
    
    return result.tolist()
