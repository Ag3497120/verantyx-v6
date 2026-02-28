def transform(grid):
    from collections import Counter, deque
    rows = len(grid)
    cols = len(grid[0])
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    result = [row[:] for row in grid]
    
    visited = [[False]*cols for _ in range(rows)]
    regions = []
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != bg:
                q = deque([(r, c)])
                visited[r][c] = True
                cells = [(r, c)]
                v = grid[r][c]
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == v:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                            cells.append((nr, nc))
                regions.append((cells, v))
    
    markers = [(cells[0], v) for cells, v in regions if len(cells) == 1]
    
    for (mr, mc), mv in markers:
        for r in range(rows):
            if r == mr and mc == mc:  # skip marker itself
                pass  # keep original
            else:
                orig = grid[r][mc]
                if orig == mv:
                    result[r][mc] = bg
                else:
                    result[r][mc] = mv
    
    return result
