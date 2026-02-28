def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    bg = 7
    result = [row[:] for row in grid]
    
    # Find non-background corners
    corners = [(0,0), (0,cols-1), (rows-1,0), (rows-1,cols-1)]
    
    for (r,c) in corners:
        v = grid[r][c]
        if v == bg: continue
        # Clear corner
        result[r][c] = bg
        
        if r == 0 and c == 0:  # top-left
            # 2x2 at (1,1),(1,2),(2,1),(2,2)
            for dr in [1,2]:
                for dc in [1,2]:
                    if r+dr < rows and c+dc < cols:
                        result[r+dr][c+dc] = v
        elif r == 0 and c == cols-1:  # top-right
            # 2x2 at (1,c-2),(1,c-1),(2,c-2),(2,c-1)
            for dr in [1,2]:
                for dc in [-2,-1]:
                    if r+dr < rows and c+dc >= 0:
                        result[r+dr][c+dc] = v
        elif r == rows-1 and c == 0:  # bottom-left
            # L-shape at (r-3,2),(r-2,2),(r-1,3)
            offsets = [(-3,2),(-2,2),(-1,3)]
            for dr,dc in offsets:
                if 0 <= r+dr < rows and 0 <= c+dc < cols:
                    result[r+dr][c+dc] = v
        elif r == rows-1 and c == cols-1:  # bottom-right
            # L-shape at (r-3,c-2),(r-2,c-2),(r-1,c-3)
            offsets = [(-3,-2),(-2,-2),(-1,-3)]
            for dr,dc in offsets:
                if 0 <= r+dr < rows and 0 <= c+dc < cols:
                    result[r+dr][c+dc] = v
    
    return result
