def transform(grid):
    import numpy as np
    from collections import deque
    g = np.array(grid)
    H, W = g.shape
    visited = np.zeros((H, W), bool)
    q = deque()
    for r in range(H):
        for c in range(W):
            if (r == 0 or r == H-1 or c == 0 or c == W-1) and g[r, c] == 0:
                visited[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and g[nr, nc] == 0:
                visited[nr, nc] = True
                q.append((nr, nc))
    out = g.copy()
    out[g == 0] = np.where(visited[g == 0], 2, 5)
    return out.tolist()
