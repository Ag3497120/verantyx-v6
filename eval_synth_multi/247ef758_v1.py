def transform(grid_list):
    import numpy as np
    from collections import Counter, defaultdict
    
    grid = np.array(grid_list)
    H, W = grid.shape
    out = grid.copy()
    
    # Find divider column
    div_col = None
    for c in range(W):
        col = grid[:, c]
        vals = set(int(v) for v in col)
        if len(vals) == 1 and 0 not in vals:
            div_col = c
            break
    if div_col is None:
        return grid_list
    
    rs = div_col + 1
    
    # Frame border default value
    border_vals = []
    border_vals.extend(int(v) for v in grid[0, rs:])
    border_vals.extend(int(v) for v in grid[H-1, rs:])
    border_vals.extend(int(v) for v in grid[:, rs])
    border_vals.extend(int(v) for v in grid[:, W-1])
    bg = Counter(border_vals).most_common(1)[0][0]
    
    # Find markers
    col_markers = defaultdict(list)
    row_markers = defaultdict(list)
    for c in range(rs, W):
        v = int(grid[0][c])
        if v != bg and v != 0:
            col_markers[v].append(c)
    for r in range(H):
        v = int(grid[r][rs])
        if v != bg and v != 0:
            row_markers[v].append(r)
    
    # Group cells by color on left side
    shapes = defaultdict(list)
    for r in range(H):
        for c in range(div_col):
            v = int(grid[r][c])
            if v != 0:
                shapes[v].append((r, c))
    
    # Place each shape at its marker intersections
    # Process in reverse marker-row order so top shapes overwrite
    shape_items = [(color, cells) for color, cells in shapes.items() 
                   if color in col_markers and color in row_markers]
    # Sort so shapes higher on left are placed LAST (and win overlaps)
    shape_items.sort(key=lambda x: -max(r for r, c in x[1]))
    
    for color, cells in shape_items:
        # Remove from left
        for r, c in cells:
            out[r][c] = 0
        
        # Find center
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        cr = (min(rows) + max(rows)) / 2.0
        cc = (min(cols) + max(cols)) / 2.0
        
        # Place at all marker intersections
        for mr in row_markers[color]:
            for mc in col_markers[color]:
                for r, c in cells:
                    nr = round(mr + (r - cr))
                    nc = round(mc + (c - cc))
                    if 0 <= nr < H and 0 <= nc < W:
                        out[nr][nc] = color
    
    return out.tolist()
