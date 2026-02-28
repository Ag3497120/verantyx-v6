from collections import deque

def transform(grid):
    h, w = len(grid), len(grid[0])
    out = [list(row) for row in grid]
    
    # BFS flood-fill from all 1-cells through 0-cells
    q = deque()
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 1:
                q.append((r, c))
    
    while q:
        r, c = q.popleft()
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<h and 0<=nc<w and out[nr][nc] == 0:
                out[nr][nc] = 1
                q.append((nr, nc))
    
    return out
