def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find column of 4s
    col4 = None
    for c in range(cols):
        if grid[0][c] == 4:
            col4 = c
            break
    
    # Find position of 5
    five_r, five_c = None, None
    for r in range(rows):
        for c in range(col4 + 1, cols):
            if grid[r][c] == 5:
                five_r, five_c = r, c - (col4 + 1)
                break
    
    out_cols = cols - (col4 + 1)
    out = [[0] * out_cols for _ in range(rows)]
    out[five_r][five_c] = 5
    
    # Parse blocks
    blocks = []
    r = 0
    while r < rows:
        has_content = any(grid[r][c] != 0 for c in range(col4))
        if has_content:
            left_color = 0
            right_color = 0
            for rr in range(r, min(r + 3, rows)):
                for c in range(3):
                    if grid[rr][c] != 0:
                        left_color = grid[rr][c]
                for c in range(4, 7):
                    if grid[rr][c] != 0:
                        right_color = grid[rr][c]
            blocks.append((left_color, right_color))
            r += 3
        else:
            r += 1
    
    color_props = {
        1: ('R', 3),
        2: ('L', 2),
        3: ('L', 4),
        6: ('D', 2),
    }
    
    segments = [b[0] for b in blocks] + [b[1] for b in blocks]
    
    cur_r = five_r
    cur_c = five_c
    
    for color in segments:
        direction, length = color_props[color]
        start_r = cur_r + 1
        start_c = cur_c
        
        if direction == 'R':
            for i in range(length):
                out[start_r][start_c + i] = color
            cur_r = start_r
            cur_c = start_c + length - 1
        elif direction == 'L':
            for i in range(length):
                out[start_r][start_c - i] = color
            cur_r = start_r
            cur_c = start_c - length + 1
        elif direction == 'D':
            for i in range(length):
                out[start_r + i][start_c] = color
            cur_r = start_r + length - 1
            cur_c = start_c
    
    return out
