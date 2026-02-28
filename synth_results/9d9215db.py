import numpy as np
from collections import defaultdict

def transform(grid):
    g = np.array(grid)
    out = np.zeros_like(g)
    h, w = g.shape
    
    # Step 1: Reflect all non-zero cells around center (4-way symmetry)
    reflected = {}  # (r, c) -> v
    for r in range(h):
        for c in range(w):
            if g[r, c] != 0:
                v = int(g[r, c])
                for nr, nc in [(r, c), (r, w-1-c), (h-1-r, c), (h-1-r, w-1-c)]:
                    reflected[(nr, nc)] = v
    
    for (r, c), v in reflected.items():
        out[r, c] = v
    
    # Collect positions per value
    val_positions = defaultdict(list)
    for (r, c), v in reflected.items():
        val_positions[v].append((r, c))
    
    # Identify corner markers (exactly 4 positions) and edge fillers (> 4)
    corner_vals = {v for v, pos in val_positions.items() if len(pos) <= 4}
    edge_vals = {v for v, pos in val_positions.items() if len(pos) > 4}
    
    # For corner values, find their positions per row and per col
    corner_row_range = {}  # row -> (min_col, max_col) from all corner values
    corner_col_range = {}  # col -> (min_row, max_row) from all corner values
    for v in corner_vals:
        for r, c in val_positions[v]:
            if r not in corner_row_range:
                corner_row_range[r] = [c, c]
            else:
                corner_row_range[r][0] = min(corner_row_range[r][0], c)
                corner_row_range[r][1] = max(corner_row_range[r][1], c)
            if c not in corner_col_range:
                corner_col_range[c] = [r, r]
            else:
                corner_col_range[c][0] = min(corner_col_range[c][0], r)
                corner_col_range[c][1] = max(corner_col_range[c][1], r)
    
    # Step 2: Fill edge values
    for v in edge_vals:
        positions = val_positions[v]
        row_cols = defaultdict(list)
        col_rows = defaultdict(list)
        for r, c in positions:
            row_cols[r].append(c)
            col_rows[c].append(r)
        
        # Horizontal fill: fill row r if edge positions are interior to corner positions
        for r, cs in row_cols.items():
            if len(cs) == 2 and r in corner_row_range:
                c1, c2 = min(cs), max(cs)
                cmin, cmax = corner_row_range[r]
                if c1 >= cmin and c2 <= cmax and c2 - c1 > 2:
                    for c in range(c1, c2 + 1, 2):
                        if out[r, c] == 0:
                            out[r, c] = v
        
        # Vertical fill: fill col c if edge positions are interior to corner positions
        for c, rs in col_rows.items():
            if len(rs) == 2 and c in corner_col_range:
                r1, r2 = min(rs), max(rs)
                rmin, rmax = corner_col_range[c]
                if r1 >= rmin and r2 <= rmax and r2 - r1 > 2:
                    for r in range(r1, r2 + 1, 2):
                        if out[r, c] == 0:
                            out[r, c] = v
    
    return out.tolist()
