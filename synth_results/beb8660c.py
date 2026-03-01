def transform(grid):
    rows = len(grid); cols = len(grid[0])
    floor_row = None
    for r in range(rows-1, -1, -1):
        if any(grid[r][c] != 0 for c in range(cols)):
            floor_row = r
            break
    objects = []
    for r in range(rows):
        if r == floor_row:
            continue
        row = grid[r]
        non_zero = [(c, v) for c, v in enumerate(row) if v != 0]
        if non_zero:
            objects.append(non_zero)
    objects.sort(key=lambda x: len(x))
    result = [[0]*cols for _ in range(rows)]
    result[floor_row] = grid[floor_row][:]
    place_row = floor_row - 1
    for obj in reversed(objects):
        n = len(obj)
        for i, (orig_c, v) in enumerate(obj):
            result[place_row][cols - n + i] = v
        place_row -= 1
    return result
