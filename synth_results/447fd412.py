def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [list(row) for row in grid]
    
    # Find the template shape (connected component containing 1s with 2 as anchor marker)
    # Find the 1s pattern - assume it's defined as the connected region of non-zero values 
    # that includes 1s, with 2s as anchor points
    
    # Find all non-zero values
    nonzero = [(r,c,grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != 0]
    
    # Count value types
    from collections import Counter
    cnt = Counter(v for r,c,v in nonzero)
    
    # The template is the shape made of 1s
    # The anchor value is 2 (appears multiple times or at special positions)
    # Find the template: the connected shape of 1s with a 2 embedded
    
    # Find the template region (connected component that includes a 2)
    template_cells = {}  # {(r,c): value}
    template_anchor = None
    
    visited = [[False]*cols for _ in range(rows)]
    from collections import deque
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                # BFS
                q = deque([(r,c)])
                visited[r][c] = True
                comp = {(r,c): grid[r][c]}
                while q:
                    rr, cc = q.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = rr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] != 0:
                            visited[nr][nc] = True
                            q.append((nr,nc))
                            comp[(nr,nc)] = grid[nr][nc]
                
                # Check if this component has 2 as anchor
                anchors_in = [(r,c) for r,c in comp if comp[(r,c)] == 2]
                if anchors_in and any(comp[(r,c)] == 1 for r,c in comp):
                    template_cells = comp
                    # Find anchor relative position
                    # Template anchor = position of the 2 within the template
                    template_anchor = anchors_in[0]
    
    if not template_cells or template_anchor is None:
        return result
    
    # Find all standalone 2s (not in template)
    standalone_twos = [(r,c) for r in range(rows) for c in range(cols) 
                       if grid[r][c] == 2 and (r,c) not in template_cells]
    
    # Place template at each standalone 2
    ar, ac = template_anchor
    for (tr, tc) in standalone_twos:
        dr = tr - ar
        dc = tc - ac
        for (r,c), v in template_cells.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                result[nr][nc] = v
    
    return result
