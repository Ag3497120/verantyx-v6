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
        # Try filtering bg
        vals = [(c, v) for c, v in enumerate(row) if v != bg]
        if len(vals) >= 3:
            gaps = [vals[i+1][0] - vals[i][0] for i in range(len(vals)-1)]
            if all(gap == 2 for gap in gaps):
                # Exclude if all values are border
                colors = [v for _, v in vals if v != bv]
                if colors:
                    chain = [v for _, v in vals]
                    chain_row = r
                    break
        # Try filtering border
        vals = [(c, v) for c, v in enumerate(row) if v != bv]
        if len(vals) >= 3:
            gaps = [vals[i+1][0] - vals[i][0] for i in range(len(vals)-1)]
            if all(gap == 2 for gap in gaps):
                chain = [v for _, v in vals]
                chain_row = r
                break
    
    if not chain:
        return grid
    
    # Find center pixels: for each box, the single pixel that makes it unique
    # In training: boxes are solid color blocks bordered by 1; the color IS the center
    # In test: boxes are 2-filled with a single colored center pixel
    # Universal: find non-border colored pixels that differ from their neighbors
    
    # Step 1: Find ALL non-border, non-bg unique colored pixels (not on chain row)
    # These are box centers
    centers = []
    for r in range(rows):
        if abs(r - chain_row) <= 1:
            continue
        for c in range(cols):
            v = int(g[r, c])
            if v != bv and v != bg:
                centers.append((r, c, v))
            elif v == bg:
                # bg-valued center: must be surrounded by non-bg non-border values
                adj_non_bg = 0
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        nv = int(g[nr, nc])
                        if nv != bg and nv != bv:
                            adj_non_bg += 1
                if adj_non_bg >= 2:
                    centers.append((r, c, v))
    
    if not centers:
        return grid
    
    # Group centers by box: centers that are adjacent belong to same box
    # Use connected components on centers
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
    
    # For each group, find bounding box and representative color
    boxes = []
    for group in box_groups:
        rs = [p[0] for p in group]
        cs = [p[1] for p in group]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        
        # Box boundary: expand outward through border_val until hitting bg or edge
        cr_mid = (min_r + max_r) // 2
        cc_mid = (min_c + max_c) // 2
        top = min_r
        while top > 0 and int(g[top-1, cc_mid]) == bv:
            top -= 1
        bot = max_r
        while bot < rows-1 and int(g[bot+1, cc_mid]) == bv:
            bot += 1
        left = min_c
        while left > 0 and int(g[cr_mid, left-1]) == bv:
            left -= 1
        right = max_c
        while right < cols-1 and int(g[cr_mid, right+1]) == bv:
            right += 1
        
        # Find center row/col
        cr = (min_r + max_r) // 2
        cc = (min_c + max_c) // 2
        color = int(g[cr, cc])
        
        # Fill rows/cols = center position
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        if h % 2 == 0:
            frows = [min_r + h//2 - 1, min_r + h//2]
        else:
            frows = [cr]
        if w % 2 == 0:
            fcols = [min_c + w//2 - 1, min_c + w//2]
        else:
            fcols = [cc]
        
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
                
                def has_between_h(br, bl, rt, rb):
                    return any(ob[0] == rt and ob[2] == rb and ob[1] > br and ob[3] < bl for ob in boxes)
                def has_between_v(bbt, bt, cl, cr):
                    return any(ob[1] == cl and ob[3] == cr and ob[0] > bbt and ob[2] < bt for ob in boxes)
                
                if t1 == t2 and bt1 == bt2:
                    if r1 < l2 and not has_between_h(r1, l2, t1, bt1):
                        for fr in frows1:
                            for fc in range(r1 + 1, l2):
                                out[fr, fc] = c1
                    elif r2 < l1 and not has_between_h(r2, l1, t1, bt1):
                        for fr in frows1:
                            for fc in range(r2 + 1, l1):
                                out[fr, fc] = c1
                elif l1 == l2 and r1 == r2:
                    if bt1 < t2 and not has_between_v(bt1, t2, l1, r1):
                        for fc in fcols1:
                            for fr in range(bt1 + 1, t2):
                                out[fr, fc] = c1
                    elif bt2 < t1 and not has_between_v(bt2, t1, l1, r1):
                        for fc in fcols1:
                            for fr in range(bt2 + 1, t1):
                                out[fr, fc] = c1
    
    return out.tolist()
