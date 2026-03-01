def transform(grid):
    import numpy as np
    g = np.array(grid)
    out = g.copy()
    rows, cols = g.shape
    
    # Find right rectangle border
    rect_left = -1
    for c in range(cols):
        if g[0, c] == 5 and g[rows-1, c] == 5:
            rect_left = c
            break
    if rect_left < 0:
        return grid
    
    rect_right = cols - 1
    int_top = 1
    int_bot = rows - 2
    int_height = int_bot - int_top + 1
    
    # Find left staircase bar row
    bar_row = -1
    max_5s = 0
    for r in range(rows):
        count = sum(1 for c in range(rect_left) if g[r, c] == 5)
        if count > max_5s:
            max_5s = count
            bar_row = r
    
    bar_left = bar_right = -1
    for c in range(rect_left):
        if g[bar_row, c] == 5:
            if bar_left < 0: bar_left = c
            bar_right = c
    
    # Step heights above bar for each left column
    staircase = {}
    for c in range(bar_left, bar_right + 1):
        h = 0
        for r in range(bar_row - 1, -1, -1):
            if g[r, c] == 5:
                h += 1
            else:
                break
        staircase[c] = h
    
    bottom_rows = int_bot - bar_row
    
    # Clear left staircase
    for r in range(rows):
        for c in range(rect_left):
            if out[r, c] == 5:
                out[r, c] = 0
    
    # Process each column in the bar range
    for lc in range(bar_left, bar_right + 1):
        rc = rect_left + 1 + lc
        if rc > rect_right - 1:
            continue
        
        is_color_col = (lc - bar_left) % 2 == 0
        sh = staircase[lc]
        total_stair = sh + 1
        
        if is_color_col:
            # Find color, input top 5s, and input color rows
            color = 0
            top_5s_input = 0
            input_color_rows = 0
            found_color = False
            for r in range(int_top, int_bot + 1):
                v = int(g[r, rc])
                if not found_color:
                    if v == 5:
                        top_5s_input += 1
                    elif v != 0:
                        color = v
                        input_color_rows += 1
                        found_color = True
                else:
                    if v == color:
                        input_color_rows += 1
                    else:
                        break
            
            available = int_height - total_stair - bottom_rows
            color_rows = min(input_color_rows, available)
            top_5s_output = available - color_rows
            
            # Place: top_5s, then color, then staircase 5s, then bottom 0s
            r = int_top
            for _ in range(top_5s_output):
                out[r, rc] = 5
                r += 1
            for _ in range(color_rows):
                out[r, rc] = color
                r += 1
            for _ in range(total_stair):
                out[r, rc] = 5
                r += 1
            for _ in range(bottom_rows):
                out[r, rc] = 0
                r += 1
        else:
            # Even column: place 5 at staircase positions, 0 elsewhere
            for r in range(int_top, int_bot + 1):
                if bar_row - sh <= r <= bar_row:
                    out[r, rc] = 5
                else:
                    out[r, rc] = 0
    
    # Place horizontal bar
    for c in range(rect_left + 1 + bar_left, rect_left + 1 + bar_right + 1):
        if rect_left < c < rect_right:
            out[bar_row, c] = 5
    
    return out.tolist()
