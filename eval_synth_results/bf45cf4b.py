def transform(grid):
    R, C = len(grid), len(grid[0])
    bg = grid[0][0]
    
    # Group non-bg cells by region (using bounding boxes of non-bg areas)
    # Find all non-bg cells
    non_bg = [(r,c) for r in range(R) for c in range(C) if grid[r][c] != bg]
    
    # Cluster into two groups based on spatial proximity
    # Find connected components using 8-connectivity on non-bg cells
    visited = [[False]*C for _ in range(R)]
    components = []
    
    def flood(r0, c0):
        stack = [(r0, c0)]
        cells = []
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= R or c < 0 or c >= C: continue
            if visited[r][c] or grid[r][c] == bg: continue
            visited[r][c] = True
            cells.append((r, c))
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    if dr == 0 and dc == 0: continue
                    stack.append((r+dr, c+dc))
        return cells
    
    for r in range(R):
        for c in range(C):
            if not visited[r][c] and grid[r][c] != bg:
                cells = flood(r, c)
                if cells:
                    components.append(cells)
    
    # Extract patterns
    patterns = []
    for cells in components:
        minr = min(r for r,c in cells)
        maxr = max(r for r,c in cells)
        minc = min(c for r,c in cells)
        maxc = max(c for r,c in cells)
        h, w = maxr - minr + 1, maxc - minc + 1
        pat = [[bg]*w for _ in range(h)]
        colors = set()
        for r, c in cells:
            pat[r-minr][c-minc] = grid[r][c]
            colors.add(grid[r][c])
        patterns.append((pat, h, w, colors))
    
    # Shape = 1 non-bg color, tile = multiple
    shape_pat = tile_pat = None
    for pat, h, w, colors in patterns:
        if len(colors) == 1:
            if shape_pat is None:
                shape_pat = (pat, h, w)
        else:
            if tile_pat is None:
                tile_pat = (pat, h, w)
    
    if shape_pat is None or tile_pat is None:
        sorted_pats = sorted(patterns, key=lambda x: len(x[3]))
        shape_pat = (sorted_pats[0][0], sorted_pats[0][1], sorted_pats[0][2])
        tile_pat = (sorted_pats[-1][0], sorted_pats[-1][1], sorted_pats[-1][2])
    
    sp, sh, sw = shape_pat
    tp, th, tw = tile_pat
    
    out_h = sh * th
    out_w = sw * tw
    out = [[bg]*out_w for _ in range(out_h)]
    
    for sr in range(sh):
        for sc in range(sw):
            if sp[sr][sc] != bg:
                for tr in range(th):
                    for tc in range(tw):
                        out[sr*th+tr][sc*tw+tc] = tp[tr][tc]
    
    return out
