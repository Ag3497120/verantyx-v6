
def transform(grid):
    import numpy as np
    from collections import defaultdict
    g = np.array(grid)
    h, w = g.shape
    bg = 0
    
    # Find connected components
    from scipy import ndimage
    colors = set(g.flatten().tolist()) - {bg}
    
    # For each color, find connected components
    labeled = {}
    for color in colors:
        mask = (g == color).astype(int)
        lbl, n = ndimage.label(mask)
        for i in range(1, n+1):
            cells = list(zip(*np.where(lbl == i)))
            labeled[id(cells)] = (color, cells)
    
    # Identify: shapes (multi-cell) and isolated markers (single-cell)
    components = []  # (color, cells) for each connected component
    for color in colors:
        mask = (g == color).astype(int)
        lbl, n = ndimage.label(mask)
        for i in range(1, n+1):
            cells = list(zip(*np.where(lbl == i)))
            components.append((color, cells))
    
    # Find isolated cells (single-cell components)
    isolated = [(color, cells[0]) for color, cells in components if len(cells) == 1]
    
    # Find multi-cell components (potential shapes)
    shapes = [(color, cells) for color, cells in components if len(cells) > 1]
    
    # For each shape, find adjacent cells of different color (internal markers)
    def find_internal_marker(shape_color, cells):
        cell_set = set(cells)
        for r, c in cells:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w:
                    if g[nr, nc] != bg and g[nr, nc] != shape_color and (nr,nc) not in cell_set:
                        return (nr, nc, int(g[nr, nc]))
        return None
    
    # Find remote anchors: isolated cells whose color has exactly one isolated occurrence
    # AND is not adjacent to a shape of different color
    remote_anchors = {}  # color -> (r, c)
    for color, (r, c) in isolated:
        # Check if it's "far" from any shape (not adjacent to any shape)
        # Actually: it's a remote anchor if its color doesn't appear in any multi-cell component
        # and is not an internal marker adjacent to a shape
        is_internal = False
        for sc, scells in shapes:
            if sc != color:
                for sr, scc in scells:
                    if abs(sr-r)<=1 and abs(scc-c)<=1 and (abs(sr-r)+abs(scc-c)==1):
                        # Adjacent to a shape of different color
                        is_internal = True
                        break
            if is_internal: break
        if not is_internal:
            remote_anchors[color] = (r, c)
    
    # Also detect internal markers: isolated cells adjacent to shapes
    internal_markers = {}  # shape -> (marker_color, marker_pos)
    for shape_color, cells in shapes:
        marker = find_internal_marker(shape_color, cells)
        if marker:
            mr, mc, mcolor = marker
            internal_markers[(shape_color, tuple(sorted(cells)))] = (mcolor, (mr, mc))
    
    out = g.copy()
    
    # For each shape with a matching remote anchor, move it
    for shape_color, cells in shapes:
        key = (shape_color, tuple(sorted(cells)))
        if key not in internal_markers:
            continue
        mcolor, (mr, mc) = internal_markers[key]
        if mcolor not in remote_anchors:
            continue
        anchor_r, anchor_c = remote_anchors[mcolor]
        
        # Find bounding box of shape
        rows = [r for r,c in cells]
        cols = [c for r,c in cells]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        
        # Find nearest corner to anchor
        corners = [(min_r, min_c), (min_r, max_c), (max_r, min_c), (max_r, max_c)]
        nearest = min(corners, key=lambda cr: abs(cr[0]-anchor_r)+abs(cr[1]-anchor_c))
        
        # Shift to align nearest corner with anchor
        dr = anchor_r - nearest[0]
        dc = anchor_c - nearest[1]
        
        # Clear original positions
        for r, c in cells:
            out[r, c] = bg
        out[mr, mc] = bg
        
        # Place at new positions
        for r, c in cells:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = shape_color
        # Move internal marker too
        nr, nc = mr+dr, mc+dc
        if 0 <= nr < h and 0 <= nc < w:
            out[nr, nc] = mcolor
    
    return out.tolist()
