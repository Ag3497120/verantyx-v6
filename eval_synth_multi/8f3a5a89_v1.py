def transform(grid):
    rows, cols = len(grid), len(grid[0])
    
    # Find source pixel (6)
    sr = sc = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 6:
                sr, sc = r, c
    
    # Flood fill from source through non-1 cells (4-connected)
    region = set()
    stack = [(sr, sc)]
    while stack:
        r, c = stack.pop()
        if (r, c) in region: continue
        if r < 0 or r >= rows or c < 0 or c >= cols: continue
        if grid[r][c] == 1: continue
        region.add((r, c))
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            stack.append((r+dr, c+dc))
    
    # Find connected components of 1-cells
    visited_1 = set()
    keep_1 = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and (r, c) not in visited_1:
                # BFS to find component
                comp = []
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if (cr, cc) in visited_1: continue
                    if cr < 0 or cr >= rows or cc < 0 or cc >= cols: continue
                    if grid[cr][cc] != 1: continue
                    visited_1.add((cr, cc))
                    comp.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        stack.append((cr+dr, cc+dc))
                # Check if any cell in comp is 8-connected to region
                touches = False
                for cr, cc in comp:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = cr+dr, cc+dc
                            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) in region:
                                touches = True; break
                        if touches: break
                    if touches: break
                if touches:
                    keep_1.update(comp)
    
    # Find exterior: flood fill from grid edge through non-region cells (4-connected)
    exterior = set()
    stack = []
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows-1 or c == 0 or c == cols-1):
                if (r, c) not in region:
                    stack.append((r, c))
    while stack:
        r, c = stack.pop()
        if (r, c) in exterior: continue
        if r < 0 or r >= rows or c < 0 or c >= cols: continue
        if (r, c) in region: continue
        exterior.add((r, c))
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            stack.append((r+dr, c+dc))
    
    # Build output
    out = [[8]*cols for _ in range(rows)]
    out[sr][sc] = 6
    
    # Place kept 1-cells
    for r, c in keep_1:
        out[r][c] = 1
    
    # 7-coating: region cells 8-connected to exterior or out-of-bounds
    for r, c in region:
        if grid[r][c] == 6: continue
        is_boundary = False
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r+dr, c+dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    is_boundary = True
                elif (nr, nc) in exterior:
                    is_boundary = True
                if is_boundary: break
            if is_boundary: break
        out[r][c] = 7 if is_boundary else 8
    
    return out
