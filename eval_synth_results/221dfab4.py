def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Count colors to find bg, blob, marker
    from collections import Counter
    color_counts = Counter()
    for r in grid:
        for c in r:
            color_counts[c] += 1
    
    sorted_colors = color_counts.most_common()
    bg = sorted_colors[0][0]
    blob_color = sorted_colors[1][0]
    marker_color = sorted_colors[2][0]
    
    # Find marker position (row and columns)
    marker_row = -1
    marker_cols = []
    for r in range(rows):
        mc = [c for c in range(cols) if grid[r][c] == marker_color]
        if mc:
            marker_row = r
            marker_cols = mc
            break
    
    # Check last row first (marker is usually at edge)
    for r in [rows-1, 0]:
        mc = [c for c in range(cols) if grid[r][c] == marker_color]
        if mc:
            marker_row = r
            marker_cols = mc
            break
    
    mc_set = set(marker_cols)
    
    # Build output
    out = [row[:] for row in grid]
    
    # Pattern at marker cols: period 6 starting from row 0
    # [3, bg, marker_color, bg, marker_color, bg]
    pattern = [3, bg, marker_color, bg, marker_color, bg]
    
    for r in range(rows):
        p = pattern[r % 6]
        
        if p == 3:
            # Marker cols -> 3, all blob cells -> 3
            for c in range(cols):
                if c in mc_set:
                    out[r][c] = 3
                elif grid[r][c] == blob_color:
                    out[r][c] = 3
        elif p == bg:
            # Marker cols -> bg, blob cells at marker cols -> bg
            for c in mc_set:
                out[r][c] = bg
        elif p == marker_color:
            # Marker cols -> marker_color
            for c in mc_set:
                out[r][c] = marker_color
    
    return out
