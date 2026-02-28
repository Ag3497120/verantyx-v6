
def transform(grid):
    import numpy as np
    g = np.array(grid)
    zeros_r, zeros_c = np.where(g == 0)
    if len(zeros_r) == 0:
        return g.tolist()
    # Get unique columns sorted
    unique_cols = sorted(set(zeros_c.tolist()))
    col_to_color = {c: i+1 for i, c in enumerate(unique_cols)}
    out = g.copy()
    for r, c in zip(zeros_r, zeros_c):
        out[r, c] = col_to_color[c]
    return out.tolist()
