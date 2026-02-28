def transform(grid):
    # Replace all 0s with 3
    return [[3 if v == 0 else v for v in row] for row in grid]
