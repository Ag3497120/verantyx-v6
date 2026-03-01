def transform(grid):
    import numpy as np
    from collections import Counter
    from scipy import ndimage
    
    g = np.array(grid)
    rows, cols = g.shape
    
    bg = Counter(g.flatten().tolist()).most_common(1)[0][0]
    one_val = 1  # frame border value
    
    # Find non-bg, non-1 cells: these are either frame centers or legend cells
    special = [(r, c, int(g[r, c])) for r in range(rows) for c in range(cols) 
               if g[r, c] != bg and g[r, c] != one_val]
    
    # Find connected components of non-bg cells
    non_bg_mask = (g != bg)
    struct = ndimage.generate_binary_structure(2, 2)
    labeled, n = ndimage.label(non_bg_mask, structure=struct)
    
    # Find which components contain colored (non-1) cells = frame components
    # vs isolated colored cells = legend
    comp_sizes = Counter(labeled.flatten().tolist())
    del comp_sizes[0]
    
    # Frames = large components with 1-borders and colored centers
    # Legend = isolated colored cells (not connected to any frame via 8-connectivity)
    
    # Find legend cells: colored cells whose 8-neighbors are all bg
    legend_cells = []
    frame_centers = []  # (val, center_row, center_col)
    
    for r, c, v in special:
        # Check if this colored cell is isolated or part of a frame
        lbl = labeled[r, c]
        # Find all cells of this component
        comp_cells = list(zip(*np.where(labeled == lbl)))
        # If the component contains only 1 colored cell type and it's isolated
        comp_vals = [g[cr, cc] for cr, cc in comp_cells]
        has_one = one_val in comp_vals
        has_bg = False
        
        if not has_one:
            # Isolated colored cell = legend
            legend_cells.append((r, c, v))
    
    # Legend: sort by column position to get order
    legend_cells.sort(key=lambda x: x[1])
    legend_order = [v for _, _, v in legend_cells]
    legend_pos = {v: i for i, v in enumerate(legend_order)}
    
    # Find frame centers: colored cells adjacent to 1-cells
    for r, c, v in special:
        # Check if surrounded by 1s
        neighbors_1 = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)] 
                          if 0<=r+dr<rows and 0<=c+dc<cols and g[r+dr,c+dc]==one_val)
        if neighbors_1 > 0:
            frame_centers.append((r, c, v))
    
    if not legend_order:
        return grid
    
    # Find distinct frame positions (row groups and col groups of centers)
    center_rows = sorted(set(r for r, c, v in frame_centers))
    center_cols = sorted(set(c for r, c, v in frame_centers))
    
    # Build frame grid: center at (r, c) → value
    frame_grid = {}
    for r, c, v in frame_centers:
        frame_grid[(r, c)] = v
    
    out = g.copy()
    
    # For each pair of adjacent frame columns: fill horizontal gap
    for i in range(len(center_rows)):
        for j in range(len(center_cols) - 1):
            cr = center_rows[i]
            cc1 = center_cols[j]
            cc2 = center_cols[j + 1]
            
            v1 = frame_grid.get((cr, cc1))
            v2 = frame_grid.get((cr, cc2))
            
            if v1 is None or v2 is None:
                continue
            
            # Check legend adjacency
            p1 = legend_pos.get(v1, -1)
            p2 = legend_pos.get(v2, -1)
            if p1 < 0 or p2 < 0:
                continue
            
            if abs(p1 - p2) == 1:
                fill_val = v1 if p1 < p2 else v2
                
                # Fill the horizontal gap between (cr, cc1) and (cr, cc2)
                # The center spans some rows around cr
                # Find the non-bg region around (cr, cc1) = center range
                # Simple: find contiguous colored rows at column cc1
                center_rows_range = set()
                for r2 in range(rows):
                    if g[r2, cc1] != bg and g[r2, cc1] != one_val:
                        center_rows_range.add(r2)
                    elif g[r2, cc1] == one_val:
                        pass  # skip frame border
                
                # Gap cols: between cc1 and cc2 (exclude the 1-cells)
                gap_cols = [c for c in range(cc1+1, cc2) 
                            if all(g[r2, c] == bg for r2 in range(rows))]
                
                if not gap_cols:
                    # Gap is the cells between (not including) cc1 and cc2
                    gap_cols = list(range(cc1+1, cc2))
                
                for r2 in center_rows_range:
                    for gc in gap_cols:
                        if out[r2, gc] == bg:
                            out[r2, gc] = fill_val
    
    # For each pair of adjacent frame rows: fill vertical gap
    for j in range(len(center_cols)):
        for i in range(len(center_rows) - 1):
            cr1 = center_rows[i]
            cr2 = center_rows[i + 1]
            cc = center_cols[j]
            
            v1 = frame_grid.get((cr1, cc))
            v2 = frame_grid.get((cr2, cc))
            
            if v1 is None or v2 is None:
                continue
            
            p1 = legend_pos.get(v1, -1)
            p2 = legend_pos.get(v2, -1)
            if p1 < 0 or p2 < 0:
                continue
            
            if abs(p1 - p2) == 1:
                fill_val = v1 if p1 < p2 else v2
                
                # Center cols around cc
                center_cols_range = set()
                for c2 in range(cols):
                    if g[cr1, c2] != bg and g[cr1, c2] != one_val:
                        center_cols_range.add(c2)
                
                gap_rows = list(range(cr1+1, cr2))
                
                for gr in gap_rows:
                    for c2 in center_cols_range:
                        if out[gr, c2] == bg:
                            out[gr, c2] = fill_val
    
    return out.tolist()
