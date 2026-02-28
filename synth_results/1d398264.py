def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    out = np.zeros((h, w), dtype=int)
    
    for y in range(h):
        for x in range(w):
            if grid[y, x] != 0:
                out[y, x] = grid[y, x]
    
    for y in range(h):
        for x in range(w):
            if grid[y, x] != 0:
                v = grid[y, x]
                left = x - 1
                right = x + 1
                up = y - 1
                down = y + 1
                
                while left >= 0 and grid[y, left] == 0:
                    out[y, left] = v
                    left -= 1
                while right < w and grid[y, right] == 0:
                    out[y, right] = v
                    right += 1
                while up >= 0 and grid[up, x] == 0:
                    out[up, x] = v
                    up -= 1
                while down < h and grid[down, x] == 0:
                    out[down, x] = v
                    down += 1
                
                for dy in [-1, 1]:
                    for dx in [-1, 1]:
                        ny, nx = y + dy, x + dx
                        while 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 0:
                            out[ny, nx] = v
                            ny += dy
                            nx += dx
    
    return out.tolist()