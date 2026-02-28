def transform(grid):
    import numpy as np
    g = np.array(grid)
    H, W = g.shape
    # Background-like values (preserved)
    preserved = {4, 7}
    # Find cycle order from corners
    corners = [g[0,0], g[0,W-1], g[H-1,W-1], g[H-1,0]]
    cycle = []
    for c in corners:
        if c not in preserved and (not cycle or cycle[-1] != c):
            cycle.append(int(c))
    # Deduplicate cycle (remove same-value repeats while keeping order)
    # Build rotation mapping: each color → previous color in cycle
    n = len(cycle)
    color_map = {cycle[i]: cycle[(i-1) % n] for i in range(n)}
    out = g.copy()
    for r in range(H):
        for c in range(W):
            v = int(g[r,c])
            if v in color_map:
                out[r,c] = color_map[v]
    return out.tolist()
