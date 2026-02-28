def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    out = grid.copy()
    
    for y in range(h):
        for x in range(w):
            if y == 0 or y == h-1 or x == 0 or x == w-1:
                continue
            if grid[y][x] == grid[y-1][x] == grid[y+1][x] == grid[y][x-1] == grid[y][x+1]:
                continue
            if grid[y][x] != grid[y-1][x] and grid[y][x] != grid[y+1][x] and \
               grid[y][x] != grid[y][x-1] and grid[y][x] != grid[y][x+1]:
                neighbors = [grid[y-1][x], grid[y+1][x], grid[y][x-1], grid[y][x+1]]
                if len(set(neighbors)) == 1:
                    out[y][x] = neighbors[0]
    return out.tolist()