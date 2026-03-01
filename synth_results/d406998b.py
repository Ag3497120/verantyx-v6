def transform(grid):
    rows = len(grid); cols = len(grid[0])
    result = [row[:] for row in grid]
    even_count = sum(1 for r in range(rows) for c in range(cols) if grid[r][c]==5 and c%2==0)
    odd_count = sum(1 for r in range(rows) for c in range(cols) if grid[r][c]==5 and c%2==1)
    convert_even = even_count > odd_count
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5 and (c%2==0) == convert_even:
                result[r][c] = 3
    return result
