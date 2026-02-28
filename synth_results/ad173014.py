def transform(grid):
    result = [[6 if v==6 else v for v in row] for row in grid]
    # 6s inside the box → 3
    result = [[3 if v==6 else v for v in row] for row in result]
    return result
