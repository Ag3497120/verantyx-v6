def transform(grid):
    from collections import deque
    rows = len(grid)
    cols = len(grid[0])
    bg = 0
    
    visited = [[False]*cols for _ in range(rows)]
    regions = []
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != bg:
                q = deque([(r, c)])
                visited[r][c] = True
                cells = [(r, c)]
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc]!=bg:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                            cells.append((nr, nc))
                regions.append(cells)
    
    result = [[bg]*cols for _ in range(rows)]
    for cells in regions:
        eights = sum(1 for r, c in cells if grid[r][c] == 8)
        if eights <= 1:
            for r, c in cells:
                result[r][c] = grid[r][c]
    
    return result
