def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    sep_col = None
    for c in range(cols):
        if all(grid[r][c] == 4 for r in range(rows)):
            sep_col = c
            break
    
    five_r, five_c = None, None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                five_r, five_c = r, c
                break
    
    out_rows = rows
    out_cols = cols - sep_col - 1
    out_five_col = five_c - sep_col - 1
    
    blocks = []
    r = 0
    while r + 2 < rows:
        has_content = any(grid[r][c] != 0 for c in range(sep_col))
        if has_content:
            left = [grid[r+j][0:3] for j in range(3)]
            right = [grid[r+j][4:sep_col] for j in range(3)]
            blocks.append((left, right))
            r += 4
        else:
            r += 1
    
    def identify_shape(pattern):
        norm = tuple(tuple(1 if v else 0 for v in row) for row in pattern)
        shapes = {
            ((1,1,0),(1,0,1),(0,1,0)): ('R', 3),
            ((1,0,1),(1,0,1),(1,1,1)): ('L', 2),
            ((1,1,1),(0,1,0),(1,0,1)): ('L', 4),
            ((1,0,1),(0,1,0),(0,1,0)): ('D', 2),
        }
        return shapes.get(norm, ('R', 1))
    
    lefts = [b[0] for b in blocks]
    rights = [b[1] for b in blocks]
    
    segments = []
    for pat in lefts + rights:
        color = max(v for row in pat for v in row)
        direction, length = identify_shape(pat)
        segments.append((color, direction, length))
    
    out = [[0]*out_cols for _ in range(out_rows)]
    out[five_r][out_five_col] = 5
    
    cur_row, cur_col = five_r, out_five_col
    
    for color, direction, length in segments:
        if direction == 'R':
            r = cur_row + 1
            for c in range(cur_col, cur_col + length):
                if 0 <= c < out_cols:
                    out[r][c] = color
            cur_row = r
            cur_col = cur_col + length - 1
        elif direction == 'L':
            r = cur_row + 1
            for c in range(cur_col - length + 1, cur_col + 1):
                if 0 <= c < out_cols:
                    out[r][c] = color
            cur_row = r
            cur_col = cur_col - length + 1
        elif direction == 'D':
            for dr in range(1, length + 1):
                if cur_row + dr < out_rows:
                    out[cur_row + dr][cur_col] = color
            cur_row = cur_row + length
    
    return out
