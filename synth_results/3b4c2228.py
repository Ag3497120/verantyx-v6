
def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    # find all 2x2 blocks of color 3
    count = 0
    visited = [[False]*cols for _ in range(rows)]
    for r in range(rows-1):
        for c in range(cols-1):
            if (grid[r][c] == 3 and grid[r][c+1] == 3 and
                grid[r+1][c] == 3 and grid[r+1][c+1] == 3 and
                not visited[r][c]):
                count += 1
                visited[r][c] = True
                visited[r][c+1] = True
                visited[r+1][c] = True
                visited[r+1][c+1] = True
    result = [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(min(count, 3)):
        result[i][i] = 1
    return result
