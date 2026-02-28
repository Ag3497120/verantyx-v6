
def transform(grid):
    import numpy as np
    from scipy import ndimage
    from collections import Counter
    g = np.array(grid)
    h, w = g.shape
    
    # Background = most common
    cnt = Counter(g.flatten().tolist())
    bg = cnt.most_common(1)[0][0]
    
    # Find 9s (anchor)
    nine_cells = set(zip(*np.where(g == 9)))
    if not nine_cells:
        return g.tolist()
    
    # Find connected 9 components
    nine_mask = (g == 9).astype(int)
    nine_lbl, n9 = ndimage.label(nine_mask)
    
    out = g.copy()
    colors = set(g.flatten().tolist()) - {bg, 9}
    
    for color in colors:
        mask = (g == color).astype(int)
        lbl, nc = ndimage.label(mask)
        for i in range(1, nc + 1):
            cells = list(zip(*np.where(lbl == i)))
            if not cells: continue
            
            # Find nearest 9 cell
            min_dist = float("inf")
            nearest_nine = None
            nearest_nine_comp = None
            for r, c in cells:
                for nr, nc2 in nine_cells:
                    d = abs(r - nr) + abs(c - nc2)
                    if d < min_dist:
                        min_dist = d
                        nearest_nine = (nr, nc2)
            
            if nearest_nine is None: continue
            nr, nc2 = nearest_nine
            
            # Determine shared row or col
            cell_rows = [r for r,c in cells]
            cell_cols = [c for r,c in cells]
            nine_comp_cells = list(zip(*np.where(nine_lbl == nine_lbl[nr, nc2])))
            nine_rows = [r for r,c in nine_comp_cells]
            nine_cols = [c for r,c in nine_comp_cells]
            
            # Check if they share rows or cols
            shared_rows = set(cell_rows) & set(nine_rows)
            shared_cols = set(cell_cols) & set(nine_cols)
            
            if shared_rows or len(cells) == 1:
                # Determine horizontal direction
                shape_center_c = sum(cell_cols) / len(cell_cols)
                nine_center_c = sum(nine_cols) / len(nine_cols)
                if shape_center_c < nine_center_c:
                    direction = "left"
                else:
                    direction = "right"
            else:
                # Determine vertical direction
                shape_center_r = sum(cell_rows) / len(cell_rows)
                nine_center_r = sum(nine_rows) / len(nine_rows)
                if shape_center_r < nine_center_r:
                    direction = "up"
                else:
                    direction = "down"
            
            # Slide to edge
            min_r, max_r = min(cell_rows), max(cell_rows)
            min_c, max_c = min(cell_cols), max(cell_cols)
            
            if direction == "left":
                shift = min_c  # move left until min_c == 0
                for r, c in cells:
                    out[r, c] = bg
                for r, c in cells:
                    out[r, c - shift] = color
            elif direction == "right":
                shift = (w - 1) - max_c
                for r, c in cells:
                    out[r, c] = bg
                for r, c in cells:
                    out[r, c + shift] = color
            elif direction == "up":
                shift = min_r
                for r, c in cells:
                    out[r, c] = bg
                for r, c in cells:
                    out[r - shift, c] = color
            elif direction == "down":
                shift = (h - 1) - max_r
                for r, c in cells:
                    out[r, c] = bg
                for r, c in cells:
                    out[r + shift, c] = color
    
    return out.tolist()
