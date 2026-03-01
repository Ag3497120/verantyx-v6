def transform(grid):
    rows = len(grid)
    groups = []
    i = 0
    while i < rows:
        color = grid[i][0]
        count = 0
        while i < rows and grid[i][0] == color:
            count += 1
            i += 1
        groups.append((color, count))
    inner = groups[-1][1]
    result_size = inner
    for color, thickness in groups[:-1]:
        result_size += 2 * thickness
    result = [[0] * result_size for _ in range(result_size)]
    offset = 0
    for color, thickness in groups:
        for r in range(offset, result_size - offset):
            for c in range(offset, result_size - offset):
                result[r][c] = color
        offset += thickness
    return result
