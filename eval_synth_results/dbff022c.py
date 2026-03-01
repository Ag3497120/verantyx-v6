def transform(grid):
    from collections import Counter, deque
    rows = len(grid); cols = len(grid[0])
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    result = [row[:] for row in grid]
    legend_map = {}
    
    def is_bg(r, c):
        return r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == bg
    
    # Find isolated solid 2-row or 2-col rectangular block = legend
    best_block = None
    for r0 in range(rows - 1):
        r1 = r0 + 1
        c = 0
        while c < cols and not best_block:
            if grid[r0][c] != bg and grid[r1][c] != bg:
                c_start = c
                while c < cols and grid[r0][c] != bg and grid[r1][c] != bg:
                    c += 1
                c_end = c
                if c_end - c_start >= 2:
                    isolated = True
                    for cr in [r0, r1]:
                        if c_start > 0 and grid[cr][c_start-1] != bg: isolated = False
                        if c_end < cols and grid[cr][c_end] != bg: isolated = False
                    if r0 > 0:
                        for cc in range(c_start, c_end):
                            if grid[r0-1][cc] != bg: isolated = False; break
                    if r1 < rows - 1:
                        for cc in range(c_start, c_end):
                            if grid[r1+1][cc] != bg: isolated = False; break
                    if isolated:
                        m = {grid[r0][cc]: grid[r1][cc] for cc in range(c_start, c_end)}
                        if len(m) >= 2:
                            best_block = m
            else:
                c += 1
        if best_block: break
    
    if not best_block:
        for c0 in range(cols - 1):
            c1 = c0 + 1
            r = 0
            while r < rows and not best_block:
                if grid[r][c0] != bg and grid[r][c1] != bg:
                    r_start = r
                    while r < rows and grid[r][c0] != bg and grid[r][c1] != bg:
                        r += 1
                    r_end = r
                    if r_end - r_start >= 2:
                        isolated = True
                        for cr in [c0, c1]:
                            if r_start > 0 and grid[r_start-1][cr] != bg: isolated = False
                            if r_end < rows and grid[r_end][cr] != bg: isolated = False
                        if c0 > 0:
                            for rr in range(r_start, r_end):
                                if grid[rr][c0-1] != bg: isolated = False; break
                        if c1 < cols - 1:
                            for rr in range(r_start, r_end):
                                if grid[rr][c1+1] != bg: isolated = False; break
                        if isolated:
                            m = {grid[rr][c0]: grid[rr][c1] for rr in range(r_start, r_end)}
                            if len(m) >= 2:
                                best_block = m
                else:
                    r += 1
            if best_block: break
    
    if best_block:
        legend_map = best_block
    
    # Find enclosed bg cells for each non-bg connected component
    # Use flood fill from outside to find non-enclosed cells
    outside = [[False]*cols for _ in range(rows)]
    q = deque()
    for r in range(rows):
        for c in [0, cols-1]:
            if grid[r][c] == bg and not outside[r][c]:
                outside[r][c] = True
                q.append((r, c))
    for c in range(cols):
        for r in [0, rows-1]:
            if grid[r][c] == bg and not outside[r][c]:
                outside[r][c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and not outside[nr][nc] and grid[nr][nc] == bg:
                outside[nr][nc] = True
                q.append((nr, nc))
    
    # Now find non-bg components and their enclosed bg cells
    visited = [[False]*cols for _ in range(rows)]
    
    def bfs_shape(sr, sc, val):
        comp = []
        stack = [(sr, sc)]
        while stack:
            r, c = stack.pop()
            if r<0 or r>=rows or c<0 or c>=cols: continue
            if visited[r][c] or grid[r][c] != val: continue
            visited[r][c] = True
            comp.append((r,c))
            for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                stack.append((r+dr, c+dc))
        return comp
    
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != bg:
                val = grid[r][c]
                comp = bfs_shape(r, c, val)
                if val not in legend_map: continue
                
                # Find enclosed bg cells: bg cells adjacent to this component that aren't outside
                # Use flood fill within the enclosed region
                adj_bg = set()
                for cr, cc in comp:
                    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == bg and not outside[nr][nc]:
                            adj_bg.add((nr, nc))
                
                # BFS to find all enclosed bg cells
                enclosed = set()
                q2 = deque(adj_bg)
                enclosed.update(adj_bg)
                while q2:
                    er, ec = q2.popleft()
                    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nr, nc = er+dr, ec+dc
                        if (0 <= nr < rows and 0 <= nc < cols and 
                            grid[nr][nc] == bg and not outside[nr][nc] and (nr,nc) not in enclosed):
                            enclosed.add((nr, nc))
                            q2.append((nr, nc))
                
                fill = legend_map[val]
                for ir, ic in enclosed:
                    result[ir][ic] = fill
    
    return result
