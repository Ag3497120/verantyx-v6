def transform(grid):
    import numpy as np
    
    grid = np.array(grid)
    h, w = grid.shape
    out = []
    
    for y in range(h):
        for x in range(w):
            if grid[y, x] == 0:
                continue
            color = grid[y, x]
            # check if it's the top-left corner of a 2x2 block of the same color
            if (x + 1 < w and y + 1 < h and
                grid[y, x + 1] == color and
                grid[y + 1, x] == color and
                grid[y + 1, x + 1] == color):
                # check if it's the top-left-most such block for this color in this connected component
                # by seeing if there's an identical 2x2 block to the left or above
                if not (x > 0 and y + 1 < h and
                        grid[y, x - 1] == color and
                        grid[y + 1, x - 1] == color):
                    if not (y > 0 and x + 1 < w and
                            grid[y - 1, x] == color and
                            grid[y - 1, x + 1] == color):
                        out.append((y, x, color))
    
    if not out:
        return [[0]]
    
    min_y = min(y for y, x, c in out)
    max_y = max(y for y, x, c in out)
    min_x = min(x for y, x, c in out)
    max_x = max(x for y, x, c in out)
    
    out_h = max_y - min_y + 1
    out_w = max_x - min_x + 1
    result = np.zeros((out_h, out_w), dtype=int)
    
    for y, x, color in out:
        # map color to output value: 6->8, others stay same? Wait, check examples.
        # In examples, output is only 0 and 8. So all detected corners become 8.
        result[y - min_y, x - min_x] = 8
    
    return result.tolist()