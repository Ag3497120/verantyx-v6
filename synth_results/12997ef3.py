import numpy as np

def transform(grid):
    arr = np.array(grid)
    rows, cols = arr.shape
    
    r1, c1 = np.where(arr == 1)
    if len(r1) == 0: return grid
    rmin, rmax = r1.min(), r1.max()
    cmin, cmax = c1.min(), c1.max()
    h = rmax - rmin + 1; w = cmax - cmin + 1
    template = np.zeros((h, w), dtype=int)
    for r, c in zip(r1, c1):
        template[r-rmin, c-cmin] = 1
    
    other_vals = sorted(set(arr.flatten().tolist()) - {0, 1})
    
    # Find positions and sort colors
    color_pos = {}
    for v in other_vals:
        rv, cv = np.where(arr == v)
        color_pos[v] = (rv.mean(), cv.mean())
    
    # Determine stacking direction based on color arrangement
    color_rows = [color_pos[v][0] for v in other_vals]
    color_cols = [color_pos[v][1] for v in other_vals]
    
    row_spread = max(color_rows) - min(color_rows) if len(color_rows) > 1 else 0
    col_spread = max(color_cols) - min(color_cols) if len(color_cols) > 1 else 0
    
    if row_spread >= col_spread:
        # Vertical stacking, sort by row
        other_vals_sorted = sorted(other_vals, key=lambda v: color_pos[v][0])
        out_rows = []
        for v in other_vals_sorted:
            colored = (template * v).tolist()
            out_rows.extend(colored)
        return out_rows
    else:
        # Horizontal stacking, sort by col
        other_vals_sorted = sorted(other_vals, key=lambda v: color_pos[v][1])
        out_cols = []
        for v in other_vals_sorted:
            colored = template * v
            out_cols.append(colored)
        return np.hstack(out_cols).tolist()
