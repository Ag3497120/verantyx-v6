def transform(grid):
    h, w = len(grid), len(grid[0])
    # Find the row with 2s
    out = [[0]*w for _ in range(h)]
    two_row = -1
    two_width = 0
    for r in range(h):
        vals = [v for v in grid[r] if v != 0]
        if vals and all(v == 2 for v in vals):
            two_row = r
            two_width = sum(1 for v in grid[r] if v == 2)
            # Copy the 2-row
            out[r] = list(grid[r])
            break
    if two_row == -1:
        return grid
    # Above: color 3, width increases by 1 per row going up
    for d in range(1, two_row + 1):
        r = two_row - d
        width = two_width + d
        for c in range(min(width, w)):
            out[r][c] = 3
    # Below: color 1, width decreases by 1 per row going down
    for d in range(1, h - two_row):
        r = two_row + d
        width = two_width - d
        if width <= 0:
            break
        for c in range(min(width, w)):
            out[r][c] = 1
    return out
