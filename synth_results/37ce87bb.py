def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    bg = 7  # background
    
    # Find non-background columns
    col_info = {}  # col -> (color, length, start_row)
    for c in range(cols):
        col_vals = [grid[r][c] for r in range(rows)]
        nonbg = [(r, v) for r, v in enumerate(col_vals) if v != bg]
        if nonbg:
            color = nonbg[0][1]
            length = len(nonbg)
            start_r = nonbg[0][0]
            col_info[c] = (color, length, start_r)
    
    if not col_info:
        return grid
    
    # Separate 2-cols and 8-cols
    cols_2 = [(c, info) for c, info in col_info.items() if info[0] == 2]
    cols_8 = [(c, info) for c, info in col_info.items() if info[0] == 8]
    
    # New column position
    max_col = max(col_info.keys())
    new_col = max_col + 2
    
    if new_col >= cols:
        return grid
    
    # Compute output length
    if cols_2:
        len_2 = cols_2[0][1][1]
        # leftmost 8-col
        leftmost_8_len = min(cols_8, key=lambda x: x[0])[1][1]
        output_len = abs(len_2 - leftmost_8_len)
    else:
        # sum of 8-col lengths
        output_len = sum(info[1] for c, info in cols_8)
    
    if output_len == 0:
        return grid
    
    result = [row[:] for row in grid]
    # Place 5s at bottom output_len rows of new_col
    for r in range(rows - output_len, rows):
        result[r][new_col] = 5
    
    return result

import json
d = json.load(open("/private/tmp/arc-agi-2/data/training/37ce87bb.json"))
for i, ex in enumerate(d['train']):
    out = transform(ex['input'])
    exp = ex['output']
    print(f"train[{i}]: {'PASS' if out == exp else 'FAIL'}")
    if out != exp:
        for r in range(len(exp)):
            if out[r] != exp[r]:
                print(f"  row {r}: got={out[r]}, exp={exp[r]}")
