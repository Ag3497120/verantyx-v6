def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    out = np.full_like(grid, 7)
    
    for y in range(h):
        for x in range(w):
            if grid[y, x] != 7:
                color = grid[y, x]
                left = x - 1
                while left >= 0 and grid[y, left] == 7:
                    left -= 1
                right = x + 1
                while right < w and grid[y, right] == 7:
                    right += 1
                up = y - 1
                while up >= 0 and grid[up, x] == 7:
                    up -= 1
                down = y + 1
                while down < h and grid[down, x] == 7:
                    down += 1
                
                if left >= 0 and right < w and grid[y, left] == grid[y, right] == color:
                    out[y, left] = color
                    out[y, right] = color
                if up >= 0 and down < h and grid[up, x] == grid[down, x] == color:
                    out[up, x] = color
                    out[down, x] = color
    return out.tolist()