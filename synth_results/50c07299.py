
def transform(grid):
    import numpy as np
    from collections import Counter
    g = np.array(grid)
    bg = Counter(g.flatten().tolist()).most_common(1)[0][0]
    # Find non-bg cells
    rows, cols = np.where(g != bg)
    if len(rows) == 0:
        return g.tolist()
    color = int(g[rows[0], cols[0]])
    # Find the anti-diagonal (r+c = constant)
    diag_sum = int(rows[0]) + int(cols[0])
    # Find cells on this anti-diagonal
    cells = [(int(r), int(c)) for r, c in zip(rows, cols) if r + c == diag_sum]
    cells.sort()
    old_count = len(cells)
    new_count = old_count + 1
    # Top of old block (smallest row)
    top_k = cells[0][0]
    # New block starts at top_k - new_count
    new_top_k = top_k - new_count
    out = g.copy()
    # Clear old cells
    for r, c in cells:
        out[r, c] = bg
    # Place new cells
    for i in range(new_count):
        r = new_top_k + i
        c = diag_sum - r
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            out[r, c] = color
    return out.tolist()
