def transform(grid):
    from collections import Counter
    rows = len(grid)
    cols = len(grid[0])
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    cnt = Counter(v for v in flat if v != bg)
    
    if len(cnt) < 2:
        return grid
    
    vals = sorted(cnt.items(), key=lambda x: -x[1])
    main_val = vals[0][0]
    ind_val = vals[1][0]
    
    main_cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == main_val]
    ind_cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == ind_val]
    
    m_rows = [r for r,c in main_cells]; m_cols = [c for r,c in main_cells]
    m_rmin, m_rmax = min(m_rows), max(m_rows)
    m_cmin, m_cmax = min(m_cols), max(m_cols)
    m_height = m_rmax - m_rmin + 1
    m_width = m_cmax - m_cmin + 1
    
    result = [row[:] for row in grid]
    for r, c in ind_cells:
        result[r][c] = bg
    
    # Try placing mirror in each of 4 directions, pick the one with enough space
    # and most empty space available
    options = []
    
    # Right of main
    space_right = cols - (m_cmax + 1)
    if space_right >= m_width:
        options.append(('right', space_right))
    # Left of main
    space_left = m_cmin
    if space_left >= m_width:
        options.append(('left', space_left))
    # Below main
    space_below = rows - (m_rmax + 1)
    if space_below >= m_height:
        options.append(('below', space_below))
    # Above main
    space_above = m_rmin
    if space_above >= m_height:
        options.append(('above', space_above))
    
    if not options:
        return result
    
    # Pick the direction with most space
    best_dir = max(options, key=lambda x: x[1])[0]
    
    if best_dir == 'right':
        offset = m_cmax + 1
        for r, c in main_cells:
            nc = offset + (m_cmax - c)
            result[r][nc] = main_val
    elif best_dir == 'left':
        offset = m_cmin - 1
        for r, c in main_cells:
            nc = offset - (c - m_cmin)
            result[r][nc] = main_val
    elif best_dir == 'below':
        offset = m_rmax + 1
        for r, c in main_cells:
            nr = offset + (m_rmax - r)
            result[nr][c] = main_val
    elif best_dir == 'above':
        offset = m_rmin - 1
        for r, c in main_cells:
            nr = offset - (r - m_rmin)
            result[nr][c] = main_val
    
    return result
