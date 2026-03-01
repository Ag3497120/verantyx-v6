
def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    bg = 0
    color_cells = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != bg:
                if v not in color_cells: color_cells[v] = []
                color_cells[v].append((r, c))
    rect_colors = {}
    for color, cells in color_cells.items():
        if len(cells) < 4: continue
        rs = [r for r, c in cells]; cs = [c for r, c in cells]
        r1, r2 = min(rs), max(rs); c1, c2 = min(cs), max(cs)
        expected = set()
        for c in range(c1, c2+1): expected.add((r1, c)); expected.add((r2, c))
        for r in range(r1, r2+1): expected.add((r, c1)); expected.add((r, c2))
        if set(cells).issubset(expected) and (r2-r1 >= 2) and (c2-c1 >= 2):
            rect_colors[color] = (r1, c1, r2, c2)
    dot_colors = {c: cells for c, cells in color_cells.items() if c not in rect_colors}
    if not rect_colors or not dot_colors: return grid
    def interior_size(bbox):
        r1, c1, r2, c2 = bbox
        return max(0, r2-r1-1) * max(0, c2-c1-1)
    sorted_rects = sorted(rect_colors.items(), key=lambda x: interior_size(x[1]))
    sorted_dots = sorted(dot_colors.items(), key=lambda x: len(x[1]))
    matched = {}
    for i, (rc, bbox) in enumerate(sorted_rects):
        if i < len(sorted_dots):
            dc, _ = sorted_dots[i]
            matched[rc] = dc
    out = [[bg]*cols for _ in range(rows)]
    for rc, (r1, c1, r2, c2) in rect_colors.items():
        for c in range(c1, c2+1): out[r1][c]=rc; out[r2][c]=rc
        for r in range(r1, r2+1): out[r][c1]=rc; out[r][c2]=rc
    for rc, (r1, c1, r2, c2) in sorted_rects:
        if rc not in matched: continue
        dc = matched[rc]
        phase = (r1+1+c1+1) % 2
        for r in range(r1+1, r2):
            for c in range(c1+1, c2):
                if (r+c) % 2 == phase:
                    out[r][c] = dc
    return out
