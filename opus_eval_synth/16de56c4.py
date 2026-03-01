def transform(grid):
    rows = len(grid)
    cols = len(grid[0])

    nonzero = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                nonzero[(r, c)] = grid[r][c]

    col_cells = {}
    for (r, c), v in nonzero.items():
        col_cells.setdefault(c, []).append((r, v))

    row_cells = {}
    for (r, c), v in nonzero.items():
        row_cells.setdefault(r, []).append((c, v))

    column_based = all(len(col_cells[c]) >= 2 for c in col_cells)

    output = [[0] * cols for _ in range(rows)]

    if column_based:
        for c, cells in col_cells.items():
            _process_line(cells, rows, output, 'col', c)
    else:
        for r, cells in row_cells.items():
            _process_line(cells, cols, output, 'row', r)

    return output


def _process_line(cells, length, output, direction, index):
    color_positions = {}
    for pos, color in cells:
        color_positions.setdefault(color, []).append(pos)

    pattern_color = None
    pattern_positions = None
    singletons = []

    for color, positions in color_positions.items():
        if len(positions) >= 2:
            pattern_color = color
            pattern_positions = sorted(positions)
        else:
            singletons.append((positions[0], color))

    if pattern_color is None:
        for pos, color in cells:
            if direction == 'row':
                output[index][pos] = color
            else:
                output[pos][index] = color
        return

    period = pattern_positions[1] - pattern_positions[0]
    start = pattern_positions[0] % period
    all_tiled = set()
    pos = start
    while pos < length:
        all_tiled.add(pos)
        pos += period

    replacement = None
    non_tiled_singletons = []
    for s_pos, s_color in singletons:
        if s_pos in all_tiled:
            replacement = (s_pos, s_color)
        else:
            non_tiled_singletons.append((s_pos, s_color))

    if replacement:
        all_positions = [p for p, _ in cells]
        range_min = min(all_positions)
        range_max = max(all_positions)
        fill_color = replacement[1]
        for p in all_tiled:
            if range_min <= p <= range_max:
                if direction == 'row':
                    output[index][p] = fill_color
                else:
                    output[p][index] = fill_color
    else:
        for p in all_tiled:
            if direction == 'row':
                output[index][p] = pattern_color
            else:
                output[p][index] = pattern_color

    for s_pos, s_color in non_tiled_singletons:
        if direction == 'row':
            output[index][s_pos] = s_color
        else:
            output[s_pos][index] = s_color
