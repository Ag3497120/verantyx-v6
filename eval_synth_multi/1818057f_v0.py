def transform(grid_list):
    grid = [row[:] for row in grid_list]
    H = len(grid)
    W = len(grid[0])
    
    # Find the non-background color (the one that forms shapes)
    # Background is the most common value
    from collections import Counter
    vals = Counter(v for row in grid for v in row)
    bg = vals.most_common(1)[0][0]
    fg = vals.most_common(2)[1][0]
    
    # Find + centers: cells where cell and all 4 neighbors = fg
    marks = set()
    for r in range(1, H-1):
        for c in range(1, W-1):
            if (grid[r][c] == fg and 
                grid[r-1][c] == fg and grid[r+1][c] == fg and
                grid[r][c-1] == fg and grid[r][c+1] == fg):
                marks.add((r,c))
                marks.add((r-1,c))
                marks.add((r+1,c))
                marks.add((r,c-1))
                marks.add((r,c+1))
    
    for r, c in marks:
        grid[r][c] = 8
    
    return grid
