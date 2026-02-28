def transform(grid):
    import numpy as np
    g = np.array(grid)
    bg = int(np.bincount(g.flatten()).argmax())
    cells = [(r,c) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r,c] != bg]
    rows = set(r for r,c in cells)
    cols = set(c for r,c in cells)
    if len(rows) >= len(cols):
        cells_sorted = sorted(cells, key=lambda x: (x[0],x[1]))
    else:
        cells_sorted = sorted(cells, key=lambda x: (x[1],x[0]))
    out = g.copy()
    for i, (r,c) in enumerate(cells_sorted):
        out[r,c] = [2,8,5][i % 3]
    return out.tolist()
