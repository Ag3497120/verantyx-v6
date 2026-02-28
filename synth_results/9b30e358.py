def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find pattern region (non-bg rows)
    non_bg_rows = [r for r in range(rows) if any(v != bg for v in grid[r])]
    bg_rows = [r for r in range(rows) if all(v == bg for v in grid[r])]
    
    if not non_bg_rows:
        return grid
    
    pattern_start = min(non_bg_rows)
    pattern_end = max(non_bg_rows)
    pattern_height = pattern_end - pattern_start + 1
    pattern = grid[pattern_start:pattern_end+1]
    
    result = [row[:] for row in grid]
    
    # Fill bg_rows (above or below) with the pattern, tiled
    # Determine if bg is above or below
    if bg_rows and max(bg_rows) < pattern_start:
        # bg is above
        for i, br in enumerate(sorted(bg_rows, reverse=True)):
            # Fill with pattern from end going up
            offset = pattern_start - 1 - br
            src_row = (pattern_height - 1 - (offset % pattern_height)) % pattern_height
            result[br] = pattern[src_row][:]
    elif bg_rows and min(bg_rows) > pattern_end:
        # bg is below - not needed based on examples
        pass
    else:
        # Mixed - just tile the pattern upward
        for br in sorted(bg_rows):
            offset = pattern_start - br
            src_row = (pattern_height - (offset % pattern_height)) % pattern_height
            result[br] = pattern[src_row][:]
    
    return result
