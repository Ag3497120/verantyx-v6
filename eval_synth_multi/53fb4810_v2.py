def transform(grid):
    import numpy as np
    g = np.array(grid)
    out = g.copy()
    rows, cols = g.shape
    bg = 8
    
    # Find cross shapes (plus signs of color 1)
    # A cross has center and 4 arms
    crosses = []
    visited = set()
    
    # Find all 1-cells
    ones = set()
    for r in range(rows):
        for c in range(cols):
            if g[r, c] == 1:
                ones.add((r, c))
    
    # Find cross centers: cells with 1 that have 1 in all 4 cardinal directions
    # Actually, find connected components of 1s that form a cross/plus shape
    from collections import deque
    used = set()
    for r, c in ones:
        if (r, c) in used:
            continue
        # BFS to find connected component
        comp = set()
        q = deque([(r, c)])
        while q:
            cr, cc = q.popleft()
            if (cr, cc) in comp:
                continue
            comp.add((cr, cc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr+dr, cc+dc
                if 0 <= nr < rows and 0 <= nc < cols and g[nr, nc] == 1 and (nr, nc) not in comp:
                    q.append((nr, nc))
        used |= comp
        
        # Find bounding box
        min_r = min(x[0] for x in comp)
        max_r = max(x[0] for x in comp)
        min_c = min(x[1] for x in comp)
        max_c = max(x[1] for x in comp)
        
        # Center is the middle
        cr = (min_r + max_r) // 2
        cc = (min_c + max_c) // 2
        crosses.append((cr, cc, comp))
    
    # For each cross, find beam seeds in each direction
    for cr, cc, comp in crosses:
        min_r = min(x[0] for x in comp)
        max_r = max(x[0] for x in comp)
        min_c = min(x[1] for x in comp)
        max_c = max(x[1] for x in comp)
        
        # Check each direction for non-bg, non-1 cells adjacent to the cross
        directions = [
            ('up', -1, 0, min_r),
            ('down', 1, 0, max_r),
            ('left', 0, -1, min_c),
            ('right', 0, 1, max_c),
        ]
        
        for name, dr, dc, edge in directions:
            # Find seed pattern beyond the cross edge
            seeds = []
            if dr != 0:  # vertical
                # Collect all columns at the edge row
                edge_cells = [(r2, c2) for r2, c2 in comp if r2 == (min_r if dr < 0 else max_r)]
                # For each column in the edge, check beyond
                for _, ec in sorted(edge_cells):
                    sr = (min_r - 1) if dr < 0 else (max_r + 1)
                    seed_col = []
                    r2 = sr
                    while 0 <= r2 < rows and g[r2, ec] != bg and g[r2, ec] != 1:
                        seed_col.append((r2, ec, int(g[r2, ec])))
                        r2 += dr
                    if seed_col:
                        seeds.append((ec, seed_col))
                
                # Extend each seed column
                for ec, seed_col in seeds:
                    pattern = [s[2] for s in seed_col]
                    if not pattern:
                        continue
                    # Extend from the end of seed to grid edge
                    start_r = seed_col[-1][0] + dr
                    r2 = start_r
                    idx = 0
                    while 0 <= r2 < rows:
                        out[r2, ec] = pattern[idx % len(pattern)]
                        r2 += dr
                        idx += 1
            else:  # horizontal
                edge_cells = [(r2, c2) for r2, c2 in comp if c2 == (min_c if dc < 0 else max_c)]
                for er, _ in sorted(edge_cells):
                    sc = (min_c - 1) if dc < 0 else (max_c + 1)
                    seed_row = []
                    c2 = sc
                    while 0 <= c2 < cols and g[er, c2] != bg and g[er, c2] != 1:
                        seed_row.append((er, c2, int(g[er, c2])))
                        c2 += dc
                    if seed_row:
                        seeds.append((er, seed_row))
                
                for er, seed_row in seeds:
                    pattern = [s[2] for s in seed_row]
                    if not pattern:
                        continue
                    start_c = seed_row[-1][1] + dc
                    c2 = start_c
                    idx = 0
                    while 0 <= c2 < cols:
                        out[er, c2] = pattern[idx % len(pattern)]
                        c2 += dc
                        idx += 1
    
    return out.tolist()
