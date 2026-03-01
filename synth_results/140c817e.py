def transform(grid):
    """
    Pattern: Find rectangular regions bounded by 3s and fill the interior with 4.
    """
    import copy
    result = copy.deepcopy(grid)
    rows = len(grid)
    cols = len(grid[0])
    
    # Find all rectangles formed by 3s and fill their interiors with 4
    # A rectangle has 3s on its borders and 0s inside
    
    # Strategy: find enclosed regions using flood fill from outside
    # Mark all cells reachable from outside as "outside"
    # Remaining 0s are "inside" and should be filled with 4
    
    visited = [[False] * cols for _ in range(rows)]
    
    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols
    
    def flood_outside(r, c):
        # BFS from border to mark all exterior 0s
        from collections import deque
        queue = deque([(r, c)])
        while queue:
            r, c = queue.popleft()
            if not is_valid(r, c) or visited[r][c] or grid[r][c] != 0:
                continue
            visited[r][c] = True
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                queue.append((r + dr, c + dc))
    
    # Flood fill from all border 0s    for i in range(rows):
        if grid[i][0] == 0:
            flood_outside(i, 0)
        if grid[i][cols - 1] == 0:
            flood_outside(i, cols - 1)
    
    for j in range(cols):
        if grid[0][j] == 0:
            flood_outside(0, j)
        if grid[rows - 1][j] == 0:
            flood_outside(rows - 1, j)
    
    # All unvisited 0s are inside, fill with 4
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and not visited[i][j]:
                result[i][j] = 4
    
    return result
