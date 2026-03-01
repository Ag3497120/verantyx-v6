def transform(grid):
    from collections import Counter
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]
    
    border_col = None
    for c in range(cols):
        if all(grid[r][c] == 2 for r in range(rows)):
            border_col = c
            break
    if border_col is None:
        return out
    
    left_width = border_col
    right_width = cols - border_col - 1
    
    for r in range(rows):
        left_pattern = []
        for d in range(1, left_width + 1):
            left_pattern.append(grid[r][border_col - d])
        
        color_counts = Counter()
        color_min_dist = {}
        for d, v in enumerate(left_pattern, 1):
            if v != 0:
                color_counts[v] += 1
                if v not in color_min_dist or d < color_min_dist[v]:
                    color_min_dist[v] = d
        
        if not color_counts:
            continue
        
        colors_by_priority = sorted(color_counts.keys(), key=lambda c: -color_min_dist[c])
        
        right_vals = [0] * right_width
        for color in colors_by_priority:
            period = color_counts[color]
            for d in range(1, right_width + 1):
                if (d - 1) % period == 0:
                    right_vals[d - 1] = color
        
        for d in range(1, right_width + 1):
            out[r][border_col + d] = right_vals[d - 1]
    
    return out
