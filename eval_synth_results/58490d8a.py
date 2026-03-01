def transform(grid):
    import numpy as np
    from scipy import ndimage
    from collections import Counter, deque
    
    g = np.array(grid)
    rows, cols = g.shape
    
    bg = Counter(g.flatten().tolist()).most_common(1)[0][0]
    
    # Find 0-region bounding box
    zero_mask = (g == 0)
    rows0, cols0 = np.where(zero_mask)
    if len(rows0) == 0:
        return grid
    r1z, r2z, c1z, c2z = rows0.min(), rows0.max(), cols0.min(), cols0.max()
    
    bbox_h = r2z - r1z + 1
    bbox_w = c2z - c1z + 1
    
    out = np.zeros((bbox_h, bbox_w), dtype=int)
    
    vals = sorted([v for v in set(g.flatten().tolist()) if v != bg and v != 0])
    
    for v in vals:
        mask_v = (g == v).astype(int)
        labeled_v, n_v = ndimage.label(mask_v)
        sizes = Counter(labeled_v.flatten().tolist())
        del sizes[0]
        
        inside = []
        outside_ns = []  # non-singleton
        outside_s_cells = []  # singleton cells
        
        for lbl, sz in sizes.items():
            cells = list(zip(*np.where(labeled_v == lbl)))
            is_inside = any(r1z <= r <= r2z and c1z <= c <= c2z for r, c in cells)
            if is_inside:
                inside.append((sz, cells))
            else:
                if sz > 1:
                    outside_ns.append(sz)
                else:
                    outside_s_cells.extend(cells)
        
        if not inside:
            continue
        
        # Marker: the inside singleton
        marker_cells = [cells for sz, cells in inside if sz == 1]
        if not marker_cells:
            marker_cells = [cells for sz, cells in inside]
        marker_row = marker_cells[0][0][0]
        rel_row = marker_row - r1z
        
        # Compute output count
        if outside_ns:
            count = len(outside_ns)
        else:
            # Cluster outside singletons using 8-connectivity
            if not outside_s_cells:
                count = 1
            else:
                visited = set()
                clusters = 0
                for cell in outside_s_cells:
                    if cell in visited:
                        continue
                    clusters += 1
                    queue = deque([cell])
                    visited.add(cell)
                    while queue:
                        r, c = queue.popleft()
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0: continue
                                nc = (r+dr, c+dc)
                                if nc in set(outside_s_cells) and nc not in visited:
                                    visited.add(nc)
                                    queue.append(nc)
                count = clusters
        
        # Place in output at row=rel_row, cols 1,3,...,2*count-1
        if 0 <= rel_row < bbox_h:
            for k in range(count):
                col = 1 + 2 * k
                if col < bbox_w:
                    out[rel_row, col] = v
    
    return out.tolist()
