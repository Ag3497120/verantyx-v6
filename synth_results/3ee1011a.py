def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find all line segments (horizontal or vertical) of non-zero values
    segments = []
    
    # Check horizontal segments
    for r in range(rows):
        c = 0
        while c < cols:
            if grid[r][c] != 0:
                color = grid[r][c]
                c2 = c
                while c2 < cols and grid[r][c2] == color:
                    c2 += 1
                length = c2 - c
                segments.append((length, color))
                c = c2
            else:
                c += 1
    
    # Check vertical segments (might overlap with horizontal - pick unique ones)
    for col in range(cols):
        r = 0
        while r < rows:
            if grid[r][col] != 0:
                color = grid[r][col]
                r2 = r
                while r2 < rows and grid[r2][col] == color:
                    r2 += 1
                length = r2 - r
                if length == 1:
                    # Single cell - check if it's isolated
                    is_isolated = True
                    for dc in [-1, 1]:
                        nc = col + dc
                        if 0 <= nc < cols and grid[r][nc] == color:
                            is_isolated = False
                    if is_isolated:
                        segments.append((1, color))
                else:
                    # Check if it's a vertical segment (not part of horizontal)
                    segments.append((length, color))
                r = r2
            else:
                r += 1
    
    if not segments:
        return [[0]*1]
    
    # Sort by length descending, keep unique (color, length) pairs
    # Actually: deduplicate by keeping one per color
    seen_colors = {}
    for length, color in segments:
        if color not in seen_colors or length > seen_colors[color]:
            seen_colors[color] = length
    
    sorted_segs = sorted(seen_colors.items(), key=lambda x: -x[1])  # (color, length), desc by length
    
    # Outermost = largest length; build nested square
    outer_size = sorted_segs[0][1]
    result = [[0]*outer_size for _ in range(outer_size)]
    
    for i, (color, length) in enumerate(sorted_segs):
        # Fill from border inward
        margin = i
        for r in range(margin, outer_size - margin):
            for c in range(margin, outer_size - margin):
                # Only fill if not yet filled by inner layers
                if result[r][c] == 0:
                    result[r][c] = color
    
    # Wait, we need to fill from outer to inner, overwriting
    # Reset and fill properly
    result = [[0]*outer_size for _ in range(outer_size)]
    for i, (color, length) in enumerate(sorted_segs):
        margin = i
        for r in range(margin, outer_size - margin):
            for c in range(margin, outer_size - margin):
                result[r][c] = color
    
    return result
