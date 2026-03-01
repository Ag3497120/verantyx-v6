def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    bg = 7
    
    # Find separator rows and cols (all 6s)
    sep_rows = [r for r in range(rows) if all(grid[r][c] == 6 for c in range(cols))]
    sep_cols = [c for c in range(cols) if all(grid[r][c] == 6 for r in range(rows))]
    
    # Get row/col ranges
    def get_ranges(seps, total):
        ranges = []
        prev = 0
        for s in seps:
            if s > prev:
                ranges.append((prev, s))
            prev = s + 1
        if prev < total:
            ranges.append((prev, total))
        return ranges
    
    row_ranges = get_ranges(sep_rows, rows)
    col_ranges = get_ranges(sep_cols, cols)
    
    # Extract panels
    panels = []
    for ri, (r0, r1) in enumerate(row_ranges):
        for ci, (c0, c1) in enumerate(col_ranges):
            panel = [row[c0:c1] for row in grid[r0:r1]]
            count = sum(1 for row in panel for c in row if c != bg)
            panels.append(((ri, ci), panel, count))
    
    n_row_panels = len(row_ranges)
    n_col_panels = len(col_ranges)
    
    if n_row_panels > 1 and n_col_panels > 1:
        # 2D -> vertical, sorted by non-bg count ascending
        panels.sort(key=lambda x: x[2])
        sorted_panels = [p for _, p, _ in panels]
        panel_w = len(sorted_panels[0][0])
        result = []
        for i, p in enumerate(sorted_panels):
            if i > 0:
                result.append([6] * panel_w)
            result.extend(p)
        return result
    
    elif n_row_panels > 1 and n_col_panels == 1:
        # Vertical -> Horizontal (reverse order)
        sorted_panels = [p for _, p, _ in reversed(panels)]
        panel_h = len(sorted_panels[0])
        result = []
        for r in range(panel_h):
            row = []
            for i, p in enumerate(sorted_panels):
                if i > 0:
                    row.append(6)
                row.extend(p[r])
            result.append(row)
        return result
    
    elif n_row_panels == 1 and n_col_panels > 1:
        # Horizontal -> Vertical (same order)
        sorted_panels = [p for _, p, _ in panels]
        panel_w = len(sorted_panels[0][0])
        result = []
        for i, p in enumerate(sorted_panels):
            if i > 0:
                result.append([6] * panel_w)
            result.extend(p)
        return result
