def transform(grid):
    """
    Find rectangles bounded by 2s, fill interiors with colors based on area.
    Small area (~12) -> 8, medium (~25-30) -> 4, large (>40) -> 3
    """
    import copy
    from collections import deque
    
    result = copy.deepcopy(grid)
    rows = len(grid)
    cols = len(grid[0])
    
    visited = [[False] * cols for _ in range(rows)]
    
    def bfs_region(start_r, start_c):
        """Find a region of 0s."""
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True
        cells = [(start_r, start_c)]
        
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not visited[nr][nc] and grid[nr][nc] == 0:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
                        cells.append((nr, nc))
        
        return cells
    
    def is_enclosed(region):
        """Check if region doesn't touch grid edges."""
        for r, c in region:
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                return False
        return True
    
    # Find all enclosed regions
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and not visited[i][j]:
                region = bfs_region(i, j)
                
                if is_enclosed(region):
                    area = len(region)
                    
                    # Determine color based on area
                    if area > 40:
                        color = 3
                    elif area > 20:
                        color = 4
                    else:
                        color = 8
                    
                    for r, c in region:
                        result[r][c] = color
    
    return result
