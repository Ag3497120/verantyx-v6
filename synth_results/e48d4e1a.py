def transform(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Count non-zero cells per column
    col_counts = {}
    col_vals = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != 0:
                col_counts[c] = col_counts.get(c, 0) + 1
                col_vals[c] = v
    
    if not col_counts: return grid
    
    # Main col = column with max row count
    main_col = max(col_counts, key=lambda c: col_counts[c])
    main_val = col_vals[main_col]
    
    # Marker col = second most common non-zero column (by count, not max)
    # Actually: marker col is the col with partial count (not full rows)
    # Shift = marker_count - 1
    # Find marker col (not main col, partial count)
    marker_count = 0
    for c, cnt in col_counts.items():
        if c != main_col and cnt > marker_count:
            marker_count = cnt
    # If we can't find a marker, default shift=0
    if marker_count == 0: marker_count = 1
    shift = marker_count - 1
    
    # New main col position
    new_col = main_col - shift
    
    # Output: only the main stripe at new_col
    result = [[0]*cols for _ in range(rows)]
    for r in range(rows):
        if 0 <= new_col < cols:
            result[r][new_col] = main_val
    return result
