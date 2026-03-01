def transform(grid):
    from collections import deque
    rows, cols = len(grid), len(grid[0])
    
    zero_cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 0]
    two_cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 2]
    
    z_min_r = min(r for r,c in zero_cells)
    z_max_r = max(r for r,c in zero_cells)
    z_min_c = min(c for r,c in zero_cells)
    z_max_c = max(c for r,c in zero_cells)
    z_h = z_max_r - z_min_r + 1
    z_w = z_max_c - z_min_c + 1
    
    shape = [[grid[r][c] for c in range(cols)] for r in range(rows)]
    for r,c in zero_cells:
        shape[r][c] = 1
    
    def can_place(tr, tc):
        for dr in range(z_h):
            for dc in range(z_w):
                r, c = tr + dr, tc + dc
                if r < 0 or r >= rows or c < 0 or c >= cols:
                    return False
                if shape[r][c] != 1 and grid[r][c] != 2:
                    return False
        return True
    
    start = (z_min_r, z_min_c)
    visited = {start: 0}
    queue = deque([(start, 0)])
    
    while queue:
        (tr, tc), dist = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = tr+dr, tc+dc
            if (nr, nc) not in visited and can_place(nr, nc):
                visited[(nr, nc)] = dist + 1
                queue.append(((nr, nc), dist + 1))
    
    best = max(visited, key=visited.get)
    
    result = [row[:] for row in grid]
    for r,c in zero_cells:
        result[r][c] = 1
    for dr in range(z_h):
        for dc in range(z_w):
            r, c = best[0] + dr, best[1] + dc
            result[r][c] = 0
    
    return result
