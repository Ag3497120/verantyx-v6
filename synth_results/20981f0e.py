def transform(grid):
    # Remove 1s (set to 0), keep everything else unchanged
    return [[0 if c == 1 else c for c in row] for row in grid]
