def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    
    # Find divider column (contains 8)
    div_col = None
    for c in range(cols):
        if any(g[r, c] == 8 for r in range(rows)):
            div_col = c
            break
    
    if div_col is None:
        return grid
    
    bg = 7
    
    # Find colored rows (non-7, non-8 values)
    colored_rows = []  # (row, color)
    for r in range(rows):
        non_bg = [g[r, c] for c in range(cols) if c != div_col and g[r, c] != bg and g[r, c] != 8 and g[r, c] != 1]
        if non_bg:
            from collections import Counter
            color = Counter(non_bg).most_common(1)[0][0]
            colored_rows.append((r, color))
    
    if not colored_rows:
        return grid
    
    # Merge consecutive same-color rows into regions
    # Each region: (first_row, last_row, color)
    regions = []
    i = 0
    while i < len(colored_rows):
        j = i
        while j+1 < len(colored_rows) and colored_rows[j+1][1] == colored_rows[i][1]:
            j += 1
        regions.append((colored_rows[i][0], colored_rows[j][0], colored_rows[i][1]))
        i = j + 1
    
    # Determine zone boundaries between regions
    # Zone for each region: fill rows around the region's colored rows
    
    # Find transition rows between adjacent regions
    transitions = []  # (row, is_separator) where separator=True means that row is all-1s
    for k in range(len(regions)-1):
        r1_end = regions[k][1]  # last row of region k
        r2_start = regions[k+1][0]  # first row of region k+1
        dist = r2_start - r1_end
        mid = (r1_end + r2_start) / 2
        if dist % 2 == 0:
            # Even distance: separator at midpoint
            transitions.append((int(mid), True, k, k+1))
        else:
            # Odd distance: transition between floor(mid) and ceil(mid)
            transitions.append((int(mid) + 1, False, k, k+1))
    
    # Assign colors to rows
    result = np.zeros_like(g)
    
    # Separator rows (original colored rows)
    colored_row_set = set(r for r, _ in colored_rows)
    
    # Transition rows
    transition_sep = {}  # row -> is_separator
    transition_zone = {}  # row -> color_above, color_below
    for row, is_sep, ki, kj in transitions:
        if is_sep:
            transition_sep[row] = True
    
    # Region zone boundaries
    # Zone for region k: from boundary_start[k] to boundary_end[k]
    zone_starts = [0] * len(regions)
    zone_ends = [rows-1] * len(regions)
    
    for k in range(len(transitions)):
        row, is_sep, ki, kj = transitions[k]
        if is_sep:
            zone_ends[ki] = row - 1
            zone_starts[kj] = row + 1
        else:
            zone_ends[ki] = row - 1
            zone_starts[kj] = row
    
    # Fill each zone
    for k, (r_start, r_end, color) in enumerate(regions):
        z_start = zone_starts[k]
        z_end = zone_ends[k]
        for r in range(z_start, z_end+1):
            if r in colored_row_set:
                result[r, :] = 1
                result[r, div_col] = 8
            else:
                result[r, :] = color
                result[r, div_col] = 1
    
    # Fill transition separator rows
    for row in transition_sep:
        result[row, :] = 1
    
    return result.tolist()
