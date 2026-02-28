def transform(grid):
    grid = [list(row) for row in grid]
    rows = len(grid)
    cols = len(grid[0])
    result = [list(row) for row in grid]
    
    # Find 3-row-high rectangles of non-zero color
    # Middle row alternates: color, 0, color, 0, ...
    visited = [[False]*cols for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                # Find horizontal run at row r
                c2 = c
                while c2 < cols and grid[r][c2] == color:
                    c2 += 1
                width = c2 - c
                if width < 2:
                    continue
                # Check 3 rows
                if r+2 < rows:
                    all_match = True
                    for rr in range(r, r+3):
                        for cc in range(c, c2):
                            if grid[rr][cc] != color:
                                all_match = False
                                break
                    if all_match:
                        # Middle row alternates
                        for cc in range(c, c2):
                            pos = cc - c
                            if pos % 2 == 1:
                                result[r+1][cc] = 0
                        for rr in range(r, r+3):
                            for cc in range(c, c2):
                                visited[rr][cc] = True
    
    return result
