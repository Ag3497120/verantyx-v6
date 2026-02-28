from collections import deque, Counter

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [list(row) for row in grid]
    
    # Find path color (minority value)
    cnt = Counter(v for row in grid for v in row)
    # path color is 2 (or the less common non-background value)
    path_color = 2
    
    visited = [[False]*cols for _ in range(rows)]
    # Mark path as visited
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == path_color:
                visited[r][c] = True
    
    def bfs(sr, sc):
        q = deque([(sr, sc)])
        visited[sr][sc] = True
        cells = [(sr, sc)]
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc]:
                    visited[nr][nc] = True
                    q.append((nr, nc))
                    cells.append((nr, nc))
        return cells
    
    regions = []
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c]:
                cells = bfs(r, c)
                regions.append(cells)
    
    if not regions:
        return result
    
    # Largest region = 3, all others = 5
    max_len = max(len(r) for r in regions)
    for region in regions:
        color = 3 if len(region) == max_len else 5
        for r, c in region:
            result[r][c] = color
    
    return result
