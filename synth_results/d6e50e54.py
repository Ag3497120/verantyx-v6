def transform(grid):
    rows = len(grid); cols = len(grid[0])
    result = [row[:] for row in grid]
    ones = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c]==1]
    if not ones: return result
    minr=min(r for r,c in ones); maxr=max(r for r,c in ones)
    minc=min(c for r,c in ones); maxc=max(c for r,c in ones)
    rect_h = maxr-minr+1; rect_w = maxc-minc+1
    for r,c in ones: result[r][c] = 2
    nines = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c]==9]
    for r,c in nines:
        result[r][c] = 7
        in_row = minr <= r <= maxr; in_col = minc <= c <= maxc
        if in_row and not in_col:
            if c < minc:
                d = minc - c; steps = d if d < rect_w else d-1
                new_r, new_c = r, c + steps
            else:
                d = c - maxc; steps = d if d < rect_w else d-1
                new_r, new_c = r, c - steps
        elif in_col and not in_row:
            if r < minr:
                d = minr - r; steps = d if d < rect_h else d-1
                new_r, new_c = r + steps, c
            else:
                d = r - maxr; steps = d if d < rect_h else d-1
                new_r, new_c = r - steps, c
        else:
            new_r, new_c = r, c
        result[new_r][new_c] = 9
    return result
