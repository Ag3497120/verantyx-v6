def transform(grid):
    h = len(grid)
    w = len(grid[0])
    # cross_color is the value at the exact center (8,8) in 0-index
    cross_color = grid[8][8]
    
    # Create output as copy of input
    output = [row[:] for row in grid]
    
    # Find all cells that are not cross_color and not zero
    for r in range(h):
        for c in range(w):
            val = grid[r][c]
            if val != 0 and val != cross_color:
                # This is a colored cell to expand
                # Expand 3x3 block centered at (r,c)
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue  # keep original color at center
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            # Only change if target cell is not part of the cross
                            # Cross cells are those where row index == 8 or col index == 8
                            if not (nr == 8 or nc == 8):
                                output[nr][nc] = cross_color
    return output