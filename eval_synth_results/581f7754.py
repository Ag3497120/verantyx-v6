def transform(grid):
    import numpy as np
    from collections import Counter
    
    g = np.array(grid)
    rows, cols = g.shape
    
    bg = Counter(g.flatten().tolist()).most_common(1)[0][0]
    
    # Find all 4s
    fours = [(r, c) for r in range(rows) for c in range(cols) if g[r, c] == 4]
    
    # Find standalone 4: 4 with no 8-neighbors that are non-background (no shape nearby)
    # 8 = border value, 4 = center value
    border_val = [v for v in set(g.flatten().tolist()) if v != bg and v != 4][0]
    
    def has_border_neighbor(r, c):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols and g[nr, nc] == border_val:
                    return True
        return False
    
    standalone_four = None
    shape_fours = []
    for r, c in fours:
        if has_border_neighbor(r, c):
            shape_fours.append((r, c))
        else:
            standalone_four = (r, c)
    
    if standalone_four is None:
        return grid
    
    target_col = standalone_four[1]
    
    out = np.full_like(g, bg)
    
    # Restore standalone 4
    out[standalone_four[0], standalone_four[1]] = 4
    
    # For each shape: erase and redraw at new horizontal position
    for sr, sc in shape_fours:
        offset = target_col - sc
        # Find all cells of this shape (connected component from (sr, sc) through 4 and border_val)
        from collections import deque
        visited = set()
        queue = deque([(sr, sc)])
        visited.add((sr, sc))
        shape_cells = []
        while queue:
            r, c = queue.popleft()
            shape_cells.append((r, c, g[r, c]))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and g[nr, nc] != bg:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        # Place at offset position
        for r, c, v in shape_cells:
            nc = c + offset
            if 0 <= r < rows and 0 <= nc < cols:
                out[r, nc] = v
    
    return out.tolist()
