def transform(grid):
    import numpy as np
    g = np.array(grid)
    out = g.copy()
    rows, cols = g.shape
    bg = 8
    bv = 1
    
    # Find chain row from bottom
    chain = []
    chain_row = -1
    for r in range(rows - 1, -1, -1):
        row = [int(g[r, c]) for c in range(cols)]
        for exclude_val in [bg, bv]:
            vals = [(c, v) for c, v in enumerate(row) if v != exclude_val]
            if len(vals) >= 3:
                gaps = [vals[i+1][0] - vals[i][0] for i in range(len(vals)-1)]
                if all(gap == 2 for gap in gaps):
                    non_bv = [v for _, v in vals if v != bv]
                    if non_bv:
                        chain = [v for _, v in vals]
                        chain_row = r
                        break
        if chain:
            break
    
    if not chain:
        return grid
    
    # Find center pixels
    centers = []
    for r in range(rows):
        if abs(r - chain_row) <= 1:
            continue
        for c in range(cols):
            v = int(g[r, c])
            if v != bv and v != bg:
                centers.append((r, c, v))
            elif v == bg:
                adj = 0
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        nv = int(g[nr, nc])
                        if nv != bg and nv != bv:
                            adj += 1
                if adj >= 2:
                    centers.append((r, c, v))
    
    if not centers:
        return grid
    
    # Group by connected component
    center_set = set((r, c) for r, c, v in centers)
    visited = set()
    box_groups = []
    for r, c, v in centers:
        if (r, c) in visited:
            continue
        group = []
        queue = [(r, c)]
        visited.add((r, c))
        while queue:
            cr, cc = queue.pop(0)
            group.append((cr, cc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr+dr, cc+dc
                if (nr, nc) in center_set and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        box_groups.append(group)
    
    # Build boxes: find center position for each group
    box_centers = []
    for group in box_groups:
        rs = [p[0] for p in group]
        cs = [p[1] for p in group]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        cr = (min_r + max_r) // 2
        cc = (min_c + max_c) // 2
        color = int(g[cr, cc])
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        frows = [min_r + h//2 - 1, min_r + h//2] if h % 2 == 0 else [cr]
        fcols = [min_c + w//2 - 1, min_c + w//2] if w % 2 == 0 else [cc]
        box_centers.append((cr, cc, color, frows, fcols))
    
    # Find grid spacing from center positions
    center_rows = sorted(set(cr for cr, cc, _, _, _ in box_centers))
    center_cols = sorted(set(cc for cr, cc, _, _, _ in box_centers))
    
    # Row spacing
    if len(center_rows) >= 2:
        row_spacings = [center_rows[i+1] - center_rows[i] for i in range(len(center_rows)-1)]
        row_spacing = min(row_spacings)
    else:
        row_spacing = rows
    
    # Col spacing
    if len(center_cols) >= 2:
        col_spacings = [center_cols[i+1] - center_cols[i] for i in range(len(center_cols)-1)]
        col_spacing = min(col_spacings)
    else:
        col_spacing = cols
    
    # Box half-dimensions (from center to edge)
    half_r = row_spacing // 2
    half_c = col_spacing // 2
    
    # Build boxes with computed boundaries
    boxes = []
    for cr, cc, color, frows, fcols in box_centers:
        top = cr - half_r
        bot = cr + half_r
        left = cc - half_c
        right = cc + half_c
        boxes.append((top, left, bot, right, color, frows, fcols))
    
    # Fill gaps
    chain_pairs = [(chain[i], chain[i+1]) for i in range(len(chain) - 1)]
    
    for c1, c2 in chain_pairs:
        boxes_c1 = [b for b in boxes if b[4] == c1]
        boxes_c2 = [b for b in boxes if b[4] == c2]
        
        for b1 in boxes_c1:
            t1, l1, bt1, r1, _, frows1, fcols1 = b1
            for b2 in boxes_c2:
                t2, l2, bt2, r2, _, frows2, fcols2 = b2
                
                def hbh(xc, mc, rt, rb):
                    return any(ob[1] > xc and ob[3] < mc and ob[0] == rt and ob[2] == rb for ob in boxes)
                def hbv(xr, mr, cl, cr):
                    return any(ob[0] > xr and ob[2] < mr and ob[1] == cl and ob[3] == cr for ob in boxes)
                
                if t1 == t2 and bt1 == bt2:
                    if r1 < l2 and not hbh(r1, l2, t1, bt1):
                        for fr in frows1:
                            for fc in range(r1 + 1, l2):
                                out[fr, fc] = c1
                    elif r2 < l1 and not hbh(r2, l1, t1, bt1):
                        for fr in frows1:
                            for fc in range(r2 + 1, l1):
                                out[fr, fc] = c1
                elif l1 == l2 and r1 == r2:
                    if bt1 < t2 and not hbv(bt1, t2, l1, r1):
                        for fc in fcols1:
                            for fr in range(bt1 + 1, t2):
                                out[fr, fc] = c1
                    elif bt2 < t1 and not hbv(bt2, t1, l1, r1):
                        for fc in fcols1:
                            for fr in range(bt2 + 1, t1):
                                out[fr, fc] = c1
    
    return out.tolist()
