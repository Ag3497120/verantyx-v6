from collections import Counter

def transform(grid):
    R, C = len(grid), len(grid[0])
    bg = 0
    divider_col = None
    for c in range(C):
        if all(grid[r][c] != bg for r in range(R)):
            divider_col = c
            break
    if divider_col is None: return grid
    frame_start = divider_col + 1
    border_color = Counter(grid[0][frame_start:]).most_common(1)[0][0]
    col_markers = {}
    for c in range(frame_start, C):
        v = grid[0][c]
        if v != bg and v != border_color:
            if v not in col_markers: col_markers[v] = []
            col_markers[v].append(c)
    row_markers = {}
    for r in range(R):
        v = grid[r][frame_start]
        if v != bg and v != border_color:
            if v not in row_markers: row_markers[v] = []
            row_markers[v].append(r)
    ws_shapes = {}
    for r in range(R):
        for c in range(divider_col):
            v = grid[r][c]
            if v != bg:
                if v not in ws_shapes: ws_shapes[v] = []
                ws_shapes[v].append((r,c))
    result = [row[:] for row in grid]
    for color, positions in ws_shapes.items():
        if color not in row_markers or color not in col_markers: continue
        for r, c in positions: result[r][c] = bg
        min_r = min(r for r,c in positions); max_r = max(r for r,c in positions)
        min_c = min(c for r,c in positions); max_c = max(c for r,c in positions)
        cr = (min_r + max_r) / 2; cc = (min_c + max_c) / 2
        for target_row in row_markers[color]:
            for target_col in col_markers[color]:
                for r, c in positions:
                    nr = target_row + r - round(cr)
                    nc = target_col + c - round(cc)
                    if 0 <= nr < R and frame_start <= nc < C and result[nr][nc] == bg:
                        result[nr][nc] = color
    return result
