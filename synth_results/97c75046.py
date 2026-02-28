def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    from collections import Counter
    from math import sqrt
    
    flat = [v for row in grid for v in row]
    cnt = Counter(flat)
    
    # Find 5
    sr, sc = None, None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                sr, sc = r, c
    
    # bg is most common non-5 non-0
    bg = [v for v,_ in cnt.most_common() if v != 5 and v != 0][0]
    zero_color = 0
    
    # Find 7-cells adjacent to 0s in at least 2 directions (concave corners)
    corners = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == bg:
                adj_dirs = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<rows and 0<=nc<cols and grid[nr][nc] == zero_color:
                        adj_dirs.append((dr,dc))
                if len(adj_dirs) >= 2:
                    corners.append((r, c))
    
    # If no concave corners, find all boundary 7-cells (adjacent to at least one 0)
    if not corners:
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == bg:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<rows and 0<=nc<cols and grid[nr][nc] == zero_color:
                            corners.append((r, c))
                            break
    
    if not corners:
        return grid
    
    # Find closest corner to 5's position
    best = min(corners, key=lambda p: sqrt((p[0]-sr)**2 + (p[1]-sc)**2))
    
    result = [row[:] for row in grid]
    result[sr][sc] = bg
    result[best[0]][best[1]] = 5
    return result
