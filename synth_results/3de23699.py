
def transform(grid):
    from collections import defaultdict
    pos = defaultdict(list)
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] != 0:
                pos[grid[r][c]].append((r,c))
    # find corner color (exactly 4 cells forming rectangle corners)
    corner_color = None
    shape_color = None
    for color, cells in pos.items():
        if len(cells) == 4:
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            if len(set(rs)) == 2 and len(set(cs)) == 2:
                corner_color = color
                break
    for color in pos:
        if color != corner_color:
            shape_color = color
            break
    if corner_color is None or shape_color is None:
        return grid
    corners = pos[corner_color]
    rs = sorted(set(p[0] for p in corners))
    cs = sorted(set(p[1] for p in corners))
    r1, r2 = rs[0], rs[1]
    c1, c2 = cs[0], cs[1]
    # interior
    result = []
    for r in range(r1+1, r2):
        row = []
        for c in range(c1+1, c2):
            v = grid[r][c]
            row.append(corner_color if v == shape_color else v)
        result.append(row)
    return result
