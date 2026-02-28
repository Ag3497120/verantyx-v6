def transform(grid):
    import numpy as np
    g = np.array(grid)
    H, W = g.shape
    from collections import deque
    # Process each "object" (non-zero, separated by 0s or background)
    # Find connected components of non-zero regions
    bg = 0  # background
    visited = np.zeros((H, W), bool)
    
    def get_ring_colors_from_border(region):
        # BFS from border to find color order (outside in)
        h, w = region.shape
        v = np.zeros((h, w), bool)
        q = deque()
        for r in range(h):
            for c in range(w):
                if r == 0 or r == h-1 or c == 0 or c == w-1:
                    if not v[r, c]:
                        v[r, c] = True; q.append((r, c))
        colors = []; seen = set()
        while q:
            r, c = q.popleft()
            col = region[r, c]
            if col not in seen:
                seen.add(col); colors.append(col)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<h and 0<=nc<w and not v[nr,nc]:
                    v[nr,nc] = True; q.append((nr,nc))
        return colors
    
    out = g.copy()
    # Find rectangular objects (regions with non-zero values, surrounded by 0s or grid edge)
    # Try simple approach: find bboxes of distinct "objects"
    # Actually, apply ring reversal globally within each bounding box
    
    # Find all objects (connected components of non-zero regions, 4-connectivity)
    from scipy.ndimage import label
    labeled, n = label(g > 0)
    for i in range(1, n+1):
        mask = labeled == i
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        r1, r2 = rows.min(), rows.max()
        c1, c2 = cols.min(), cols.max()
        region = g[r1:r2+1, c1:c2+1]
        colors = get_ring_colors_from_border(region)
        reversed_colors = colors[::-1]
        cmap = {c: reversed_colors[j] for j, c in enumerate(colors)}
        sub = region.copy()
        for old, new in cmap.items():
            sub[region == old] = new
        out[r1:r2+1, c1:c2+1] = sub
    return out.tolist()
