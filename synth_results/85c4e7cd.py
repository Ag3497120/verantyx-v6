def transform(grid):
    import numpy as np
    g = np.array(grid)
    # Find concentric rings - colors ordered from outside to inside
    # by checking the "layers" using erosion
    H, W = g.shape
    # Get unique colors in order of appearance from outside to inside
    # Use distance from border to identify rings
    def get_ring_order(g):
        visited = np.zeros_like(g, dtype=bool)
        from collections import deque
        # BFS from border to find color order
        q = deque()
        for r in range(H):
            for c in range(W):
                if r == 0 or r == H-1 or c == 0 or c == W-1:
                    if not visited[r, c]:
                        visited[r, c] = True
                        q.append((r, c))
        colors = []
        seen = set()
        while q:
            r, c = q.popleft()
            col = g[r, c]
            if col not in seen:
                seen.add(col)
                colors.append(col)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                    visited[nr, nc] = True
                    q.append((nr, nc))
        # Add any remaining interior colors
        interior = g[~visited]
        for v in np.unique(interior):
            if v not in seen:
                colors.append(v)
        return colors
    
    colors = get_ring_order(g)
    reversed_colors = colors[::-1]
    color_map = {c: reversed_colors[i] for i, c in enumerate(colors)}
    out = np.vectorize(lambda x: color_map.get(x, x))(g)
    return out.tolist()
