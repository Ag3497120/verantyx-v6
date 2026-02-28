def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    out = grid.copy()
    
    for y in range(h):
        for x in range(w):
            val = grid[y, x]
            if val == 0 or val > 4:
                continue
            if val == 1:
                target = 2
            elif val == 2:
                target = 4
            elif val == 3:
                target = 6
            elif val == 4:
                target = 8
            else:
                continue
            
            # Flood fill
            stack = [(y, x)]
            visited = set()
            while stack:
                cy, cx = stack.pop()
                if (cy, cx) in visited:
                    continue
                visited.add((cy, cx))
                if 0 <= cy < h and 0 <= cx < w and grid[cy, cx] == val:
                    out[cy, cx] = target
                    for dy, dx in [(1,0),(-1,0),(0,1),(0,-1)]:
                        ny, nx = cy + dy, cx + dx
                        stack.append((ny, nx))
    return out.tolist()