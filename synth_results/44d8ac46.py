from collections import deque

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [list(row) for row in grid]
    
    # Find enclosed 0-regions
    visited = [[False]*cols for _ in range(rows)]
    q = deque()
    
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows-1 or c == 0 or c == cols-1) and grid[r][c] == 0:
                if not visited[r][c]:
                    visited[r][c] = True
                    q.append((r, c))
    
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == 0:
                visited[nr][nc] = True
                q.append((nr, nc))
    
    seen = [[False]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and not visited[r][c] and not seen[r][c]:
                q2 = deque([(r, c)])
                seen[r][c] = True
                region = set([(r, c)])
                while q2:
                    rr, cc = q2.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = rr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and not seen[nr][nc] and grid[nr][nc] == 0 and not visited[nr][nc]:
                            seen[nr][nc] = True
                            q2.append((nr, nc))
                            region.add((nr, nc))
                
                r1 = min(rr for rr,cc in region)
                r2 = max(rr for rr,cc in region)
                c1 = min(cc for rr,cc in region)
                c2 = max(cc for rr,cc in region)
                h = r2 - r1 + 1
                w = c2 - c1 + 1
                # Must be a solid rectangle (no holes) AND square
                if h == w and len(region) == h * w:
                    for rr, cc in region:
                        result[rr][cc] = 2
    
    return result
