def transform(grid):
    rows, cols = len(grid), len(grid[0])
    # Remove all-zero rows
    non_zero_rows = [grid[r] for r in range(rows) if any(v != 0 for v in grid[r])]
    if not non_zero_rows:
        return grid
    # Remove all-zero cols
    non_zero_cols = [c for c in range(cols) if any(non_zero_rows[r][c] != 0 for r in range(len(non_zero_rows)))]
    result = [[non_zero_rows[r][c] for c in non_zero_cols] for r in range(len(non_zero_rows))]
    return result
