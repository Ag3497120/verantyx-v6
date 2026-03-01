def transform(grid):
    """
    Pattern: Create a 3x3 tiling where each tile position is determined by the input.
    If input[i][j] is non-zero, place the entire input pattern in that tile position.
    If input[i][j] is zero, place all zeros in that tile position.
    """
    n = len(grid)
    m = len(grid[0])
    result = [[0] * (n * m) for _ in range(n * m)]
    
    for i in range(n):
        for j in range(m):
            if grid[i][j] != 0:
                # Place the entire input grid in this block
                for di in range(n):
                    for dj in range(m):
                        result[i * n + di][j * m + dj] = grid[di][dj]
    
    return result
