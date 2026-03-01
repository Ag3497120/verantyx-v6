
def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find separator rows (all-zero) and separator cols (all-zero)
    sep_rows = [r for r in range(rows) if all(grid[r][c] == 0 for c in range(cols))]
    sep_cols = [c for c in range(cols) if all(grid[r][c] == 0 for r in range(rows))]
    
    # Get cell row/col ranges
    def get_ranges(seps, total):
        ranges = []
        prev = -1
        for s in seps:
            if prev + 1 < s:
                ranges.append((prev+1, s-1))
            prev = s
        if prev + 1 < total:
            ranges.append((prev+1, total-1))
        return ranges
    
    row_ranges = get_ranges(sep_rows, rows)
    col_ranges = get_ranges(sep_cols, cols)
    
    # Extract each box: its border color and inner pattern
    boxes = {}  # (ri, ci) -> (border_color, inner)
    for ri, (r0, r1) in enumerate(row_ranges):
        for ci, (c0, c1) in enumerate(col_ranges):
            # Border is outermost ring
            border_color = grid[r0][c0]
            if border_color == 0:
                continue
            # Inner region
            inner = []
            for r in range(r0+1, r1):
                row = []
                for c in range(c0+1, c1):
                    row.append(grid[r][c])
                inner.append(row)
            boxes[(ri, ci)] = (border_color, inner)
    
    # Group by border color
    by_color = {}
    for (ri, ci), (bc, inner) in boxes.items():
        if bc not in by_color:
            by_color[bc] = []
        by_color[bc].append((ri, ci, inner))
    
    # Find template (non-empty) inner per color
    templates = {}
    for bc, items in by_color.items():
        for ri, ci, inner in items:
            if any(v != 0 for row in inner for v in row):
                # Check if it has the border color inside (true template)
                if any(v == bc for row in inner for v in row):
                    templates[bc] = inner
                    break
        if bc not in templates:
            # Use any non-empty
            for ri, ci, inner in items:
                if any(v != 0 for row in inner for v in row):
                    templates[bc] = inner
                    break
    
    # Build output
    result = [row[:] for row in grid]
    
    for ri, (r0, r1) in enumerate(row_ranges):
        for ci, (c0, c1) in enumerate(col_ranges):
            if (ri, ci) not in boxes:
                continue
            bc, inner = boxes[(ri, ci)]
            if bc not in templates:
                continue
            template = templates[bc]
            # Check if box is empty
            if all(v == 0 for row in inner for v in row):
                # Fill with template
                for dr, row in enumerate(template):
                    for dc, val in enumerate(row):
                        result[r0+1+dr][c0+1+dc] = val
    
    return result
