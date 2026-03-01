import numpy as np
from collections import Counter

def transform(grid):
    g = np.array(grid)
    H, W = g.shape
    out = g.copy()
    
    dom = Counter(g.flatten().tolist()).most_common(1)[0][0]
    
    # Find connected components of non-dominant cells
    visited = np.zeros((H, W), dtype=bool)
    components = []
    for r in range(H):
        for c in range(W):
            if visited[r, c] or g[r, c] == dom:
                continue
            stack = [(r, c)]
            cells = []
            while stack:
                cr, cc = stack.pop()
                if cr < 0 or cc < 0 or cr >= H or cc >= W: continue
                if visited[cr, cc] or g[cr, cc] == dom: continue
                visited[cr, cc] = True
                cells.append((cr, cc, int(g[cr, cc])))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    stack.append((cr+dr, cc+dc))
            components.append(cells)
    
    # Stamp = largest component. Anchor = all small (single-cell) components.
    components.sort(key=lambda c: len(c), reverse=True)
    stamp_comp = components[0]
    anchor_cells_all = []
    for comp in components[1:]:
        for r, c, v in comp:
            anchor_cells_all.append((r, c, v))
    
    stamp_color = stamp_comp[0][2]
    stamp_cells = [(r, c) for r, c, v in stamp_comp]
    
    # Find anchor color (most common among anchor cells)
    anchor_colors = Counter(v for r, c, v in anchor_cells_all)
    anchor_color = anchor_colors.most_common(1)[0][0]
    
    # Find anchor center (the cell with stamp_color among anchor cells)
    anchor_center = None
    for r, c, v in anchor_cells_all:
        if v == stamp_color:
            anchor_center = (r, c)
    
    all_anchor_pos = [(r, c) for r, c, v in anchor_cells_all]
    
    if anchor_center is None:
        # Use median
        rs = [r for r, c in all_anchor_pos]
        cs = [c for r, c in all_anchor_pos]
        anchor_center = (sorted(rs)[len(rs)//2], sorted(cs)[len(cs)//2])
    
    # Stamp bounding box
    sr = [r for r, c in stamp_cells]
    sc = [c for r, c in stamp_cells]
    stamp_r0, stamp_c0 = min(sr), min(sc)
    stamp_h = max(sr) - stamp_r0 + 1
    stamp_w = max(sc) - stamp_c0 + 1
    stamp_shape = set((r - stamp_r0, c - stamp_c0) for r, c in stamp_cells)
    
    # Anchor spacing
    anchor_only = [(r, c) for r, c, v in anchor_cells_all if v == anchor_color]
    dists = set()
    for i in range(len(anchor_only)):
        for j in range(i+1, len(anchor_only)):
            dr = abs(anchor_only[i][0] - anchor_only[j][0])
            dc = abs(anchor_only[i][1] - anchor_only[j][1])
            if dr > 0: dists.add(dr)
            if dc > 0: dists.add(dc)
    anchor_spacing = min(dists) if dists else 2
    
    # Offsets relative to anchor center
    cr, cc = anchor_center
    offsets = []
    for r, c in all_anchor_pos:
        dr = round((r - cr) / anchor_spacing)
        dc = round((c - cc) / anchor_spacing)
        offsets.append((dr, dc))
    
    # Clear original stamp → anchor color
    for r, c in stamp_cells:
        out[r, c] = anchor_color
    
    # Clear anchor cells → dominant
    for r, c in all_anchor_pos:
        out[r, c] = dom
    
    # Place stamp copies
    for dr, dc in offsets:
        if dr == 0 and dc == 0:
            continue
        copy_r0 = stamp_r0 + dr * stamp_h
        copy_c0 = stamp_c0 + dc * stamp_w
        for sr_off, sc_off in stamp_shape:
            r = copy_r0 + sr_off
            c = copy_c0 + sc_off
            if 0 <= r < H and 0 <= c < W:
                out[r, c] = stamp_color
    
    return out.tolist()
