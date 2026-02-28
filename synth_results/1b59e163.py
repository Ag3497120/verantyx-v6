def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    out = np.full((h, w), 4, dtype=int)
    
    for y in range(h):
        for x in range(w):
            if grid[y, x] != 4:
                if grid[y, x] == 5:
                    out[y, x] = 5
                elif grid[y, x] == 6:
                    out[y, x] = 6
                elif grid[y, x] == 8:
                    out[y, x] = 8
                else:
                    pattern = None
                    if y + 2 < h and x + 2 < w:
                        if (grid[y, x] == 1 and grid[y, x+1] == 1 and grid[y, x+2] == 1 and
                            grid[y+1, x] == 1 and grid[y+1, x+2] == 1 and
                            grid[y+2, x] == 1 and grid[y+2, x+1] == 1 and grid[y+2, x+2] == 1):
                            pattern = 'box'
                    if pattern == 'box':
                        out[y:y+3, x:x+3] = [[1,1,1],[1,2,1],[1,1,1]]
                    else:
                        if grid[y, x] == 1:
                            if y + 2 < h and grid[y+1, x] == 1 and grid[y+2, x] == 1:
                                out[y:y+3, x] = [1,1,1]
                            elif x + 2 < w and grid[y, x+1] == 1 and grid[y, x+2] == 1:
                                out[y, x:x+3] = [1,1,1]
                            elif y + 2 < h and x + 1 < w and grid[y+1, x+1] == 1 and grid[y+2, x] == 1:
                                out[y, x] = 1
                                out[y+1, x+1] = 1
                                out[y+2, x] = 1
                            else:
                                out[y, x] = 1
                        elif grid[y, x] == 2:
                            out[y, x] = 2
                        elif grid[y, x] == 3:
                            out[y, x] = 3
    return out.tolist()