
def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    bg = 0
    # Find the column of 7s (vertical line)
    col_count = {}
    color = 7
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                color = grid[r][c]
                col_count[c] = col_count.get(c, 0) + 1
    # The main column is the one with most cells
    main_col = max(col_count, key=lambda c: col_count[c])
    # Find the rows with color cells in main_col
    colored_rows = [r for r in range(rows) if grid[r][main_col] == color]
    if not colored_rows:
        return grid
    bottom_row = max(colored_rows)
    out = [[bg]*cols for _ in range(rows)]
    for row in colored_rows:
        d = bottom_row - row
        for j in range(cols):
            if abs(j - main_col) <= d:
                dist = abs(j - main_col)
                if dist % 2 == 0:
                    out[row][j] = color
                else:
                    out[row][j] = 8
    return out
