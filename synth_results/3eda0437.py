def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [list(row) for row in grid]
    
    # Find largest all-zeros rectangle with height >= 2
    heights = [0] * cols
    best_area = 0
    best_rect = None
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                heights[c] += 1
            else:
                heights[c] = 0
        
        # Find max rectangle in histogram at this row with height >= 2
        stack = []
        for c in range(cols + 1):
            h = heights[c] if c < cols else 0
            start = c
            while stack and stack[-1][1] > h:
                sc, sh = stack.pop()
                if sh >= 2:  # height constraint
                    area = sh * (c - sc)
                    if area > best_area:
                        best_area = area
                        best_rect = (r - sh + 1, sc, r, c - 1)
                start = sc
            stack.append((start, h))
    
    if best_rect:
        r1, c1, r2, c2 = best_rect
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                result[r][c] = 6
    
    return result
