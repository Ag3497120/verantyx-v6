
def transform(grid):
    import numpy as np
    from collections import Counter
    g = np.array(grid)
    # Find the 5-cross: L-shape. Find the corner (intersection of horiz and vert arms)
    fives = list(zip(*np.where(g == 5)))
    five_rows = Counter(r for r,c in fives)
    five_cols = Counter(c for r,c in fives)
    horiz_row = five_rows.most_common(1)[0][0]
    vert_col = five_cols.most_common(1)[0][0]
    # Enclosed region: rows <= horiz_row and cols <= vert_col
    # Find shapes (non-0, non-5)
    colors = set(g.flatten().tolist()) - {0, 5}
    shape_structs = {}  # color -> relative shape
    for color in colors:
        cells = sorted(zip(*np.where(g == color)))
        if not cells: continue
        rmin = min(r for r,c in cells)
        cmin = min(c for r,c in cells)
        rel = tuple(sorted((r-rmin, c-cmin) for r,c in cells))
        shape_structs[color] = rel
    # Find pair with same structure
    from collections import defaultdict
    struct_to_colors = defaultdict(list)
    for c, s in shape_structs.items():
        struct_to_colors[s].append(c)
    out = g.copy()
    for s, colors_list in struct_to_colors.items():
        if len(colors_list) >= 2:
            # Find which one is outside the enclosed region
            for color in colors_list:
                cells = list(zip(*np.where(g == color)))
                # Check if all cells are inside (rows<=horiz_row and cols<=vert_col)
                inside = all(r <= horiz_row and c <= vert_col for r,c in cells)
                if not inside:
                    out[g == color] = 5
            break
    return out.tolist()
