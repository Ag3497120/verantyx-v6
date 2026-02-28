def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    result = np.zeros_like(g)
    
    # Find divider rows/cols (all 5)
    div_rows = [r for r in range(rows) if all(g[r, c] == 5 for c in range(cols))]
    div_cols = [c for c in range(cols) if all(g[r, c] == 5 for r in range(rows))]
    
    # Mark dividers in result
    for r in div_rows:
        result[r, :] = 5
    for c in div_cols:
        result[:, c] = 5
    
    # Find room boundaries
    row_ranges = []
    prev = 0
    for dr in div_rows + [rows]:
        if dr > prev:
            row_ranges.append((prev, dr-1))
        prev = dr + 1
    
    col_ranges = []
    prev = 0
    for dc in div_cols + [cols]:
        if dc > prev:
            col_ranges.append((prev, dc-1))
        prev = dc + 1
    
    # Find non-5, non-0 color (the fill color)
    fill_color = None
    for r in range(rows):
        for c in range(cols):
            if g[r, c] != 0 and g[r, c] != 5:
                fill_color = g[r, c]
                break
        if fill_color:
            break
    
    if fill_color is None:
        return grid
    
    # Count fill_color cells per room
    rooms = []
    for r1, r2 in row_ranges:
        for c1, c2 in col_ranges:
            count = np.sum(g[r1:r2+1, c1:c2+1] == fill_color)
            rooms.append((count, r1, r2, c1, c2))
    
    max_count = max(r[0] for r in rooms)
    
    for count, r1, r2, c1, c2 in rooms:
        if count == max_count:
            result[r1:r2+1, c1:c2+1] = fill_color
    
    return result.tolist()
