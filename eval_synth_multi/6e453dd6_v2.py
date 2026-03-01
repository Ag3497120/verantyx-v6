def transform(grid):
    import numpy as np
    from collections import deque
    g = np.array(grid, dtype=int)
    rows, cols = g.shape
    
    # Find the vertical divider line (color 5)
    div_col = None
    for c in range(cols):
        if all(g[r][c] == 5 for r in range(rows)):
            div_col = c
            break
    if div_col is None:
        return grid
    
    bg = 6
    shape_color = 0
    
    # Find connected components of shape_color on the left side
    visited = set()
    shapes = []
    for r in range(rows):
        for c in range(div_col):
            if g[r][c] == shape_color and (r,c) not in visited:
                cells = []
                q = deque([(r,c)])
                visited.add((r,c))
                while q:
                    cr,cc = q.popleft()
                    cells.append((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc = cr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<div_col and (nr,nc) not in visited and g[nr][nc] == shape_color:
                            visited.add((nr,nc))
                            q.append((nr,nc))
                shapes.append(cells)
    
    out = np.full_like(g, bg)
    for r in range(rows):
        out[r][div_col] = 5
    
    for cells in shapes:
        max_c = max(c for r,c in cells)
        shift = (div_col - 1) - max_c
        
        # Place shifted shape
        for r,c in cells:
            out[r][c + shift] = shape_color
        
        # Check each row for gap condition
        from collections import defaultdict
        row_cols = defaultdict(list)
        for r,c in cells:
            row_cols[r].append(c + shift)
        
        for r, col_list in row_cols.items():
            col_list_sorted = sorted(col_list)
            rightmost = col_list_sorted[-1]
            leftmost = col_list_sorted[0]
            
            # Check if rightmost touches divider
            if rightmost != div_col - 1:
                continue
            
            # Check for gaps
            has_gap = False
            for i in range(len(col_list_sorted) - 1):
                if col_list_sorted[i+1] - col_list_sorted[i] > 1:
                    has_gap = True
                    break
            
            if has_gap:
                for c in range(div_col + 1, cols):
                    out[r][c] = 2
    
    return out.tolist()
