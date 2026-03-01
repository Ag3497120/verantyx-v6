
def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] != bg]
    if not cells:
        return grid
    color = grid[cells[0][0]][cells[0][1]]
    count = len(cells)
    min_r = min(r for r,c in cells)
    max_r = max(r for r,c in cells)
    min_c = min(c for r,c in cells)
    max_c = max(c for r,c in cells)
    center_col = (min_c + max_c) // 2
    if count >= cols:
        target_row = max_r
    else:
        target_row = (min_r + max_r) // 2
    start_c = center_col - count // 2
    start_c = max(0, min(start_c, cols - count))
    out = [[bg]*cols for _ in range(rows)]
    for c in range(start_c, start_c + count):
        if 0 <= c < cols:
            out[target_row][c] = color
    return out
