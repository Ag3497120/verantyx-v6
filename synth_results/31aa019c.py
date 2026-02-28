def transform(grid):
    rows, cols = len(grid), len(grid[0])
    
    # Count occurrences of each value
    counts = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            counts[v] = counts.get(v, 0) + 1
    
    # Find the unique value (appears exactly once)
    anchor_val = None
    anchor_r, anchor_c = 0, 0
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if counts[v] == 1:
                anchor_val = v
                anchor_r, anchor_c = r, c
                break
        if anchor_val is not None:
            break
    
    if anchor_val is None:
        return grid
    
    result = [[0]*cols for _ in range(rows)]
    # Place 3x3 ring of 2s around anchor
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            nr, nc = anchor_r + dr, anchor_c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                result[nr][nc] = 2
    result[anchor_r][anchor_c] = anchor_val
    
    return result
