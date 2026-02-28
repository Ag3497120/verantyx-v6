def transform(grid):
    import numpy as np
    g = np.array(grid)
    R, C = g.shape
    shape_color = 1
    shape_cells = [(r, c) for r in range(R) for c in range(C) if g[r, c] == shape_color]
    markers = [(r, c, int(g[r, c])) for r in range(R) for c in range(C) if g[r, c] != 0 and g[r, c] != shape_color]
    if not markers or not shape_cells: return grid
    
    out = g.copy()
    for mr, mc, mk in markers:
        # Find shape cell sharing row or col
        align = None
        for sr, sc in shape_cells:
            if sc == mc:
                align = (sr, sc, 'row'); break
            elif sr == mr:
                align = (sr, sc, 'col'); break
        if not align:
            dists = [(abs(mr-sr)+abs(mc-sc), 'row' if abs(mc-sc)<=abs(mr-sr) else 'col', sr, sc) for sr,sc in shape_cells]
            dists.sort(); _, axis, sr, sc = dists[0]
            align = (sr, sc, axis)
        ar, ac, axis = align
        if axis == 'row':
            s = ar + mr
            for sr, sc in shape_cells:
                nr = s - sr
                if 0 <= nr < R: out[nr, sc] = mk
        else:
            s = ac + mc
            for sr, sc in shape_cells:
                nc = s - sc
                if 0 <= nc < C: out[sr, nc] = mk
    return out.tolist()
