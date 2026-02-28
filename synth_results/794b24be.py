def transform(grid):
    count = sum(v for row in grid for v in row if v == 1)
    # Order: row 0 left to right, then row 1 center first
    order = [(0,0),(0,1),(0,2),(1,1),(1,0),(1,2),(2,0),(2,1),(2,2)]
    result = [[0]*3 for _ in range(3)]
    for i in range(min(count, len(order))):
        r, c = order[i]
        result[r][c] = 2
    return result