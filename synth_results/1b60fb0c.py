def transform(grid):
    h = len(grid)
    w = len(grid[0])
    
    # Create output grid, initially same as input
    output = [row[:] for row in grid]
    
    # Find all border zeros
    border_zeros = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 and (r == 0 or r == h-1 or c == 0 or c == w-1):
                border_zeros.append((r, c))
    
    # BFS to mark all exterior zeros
    visited = [[False] * w for _ in range(h)]
    from collections import deque
    q = deque(border_zeros)
    for r, c in border_zeros:
        visited[r][c] = True
    
    dirs = [(0,1),(0,-1),(1,0),(-1,0)]
    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == 0:
                visited[nr][nc] = True
                q.append((nr, nc))
    
    # Change unvisited zeros (interior holes) to 2
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 and not visited[r][c]:
                output[r][c] = 2
    
    return output