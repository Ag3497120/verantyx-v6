from collections import Counter

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    divider_col = -1
    best_count = 0
    for c in range(cols):
        vals = [grid[r][c] for r in range(rows)]
        cnt = Counter(v for v in vals if v != 0)
        if cnt:
            most_common_val, most_common_cnt = cnt.most_common(1)[0]
            if most_common_cnt >= rows - 2 and most_common_cnt > best_count:
                divider_col = c
                best_count = most_common_cnt
    
    left_has = False
    right_has = False
    for r in range(rows):
        if grid[r][divider_col] == 0:
            continue
        for c in range(divider_col):
            if grid[r][c] != 0:
                left_has = True
        for c in range(divider_col + 1, cols):
            if grid[r][c] != 0:
                right_has = True
    
    result = [row[:] for row in grid]
    
    if left_has and not right_has:
        for r in range(rows):
            if grid[r][divider_col] == 0:
                continue
            pattern = [grid[r][c] for c in range(divider_col)]
            colors = {}
            max_pos = {}
            for c_idx, v in enumerate(pattern):
                if v != 0:
                    colors[v] = colors.get(v, 0) + 1
                    max_pos[v] = max(max_pos.get(v, -1), c_idx)
            if not colors:
                continue
            for i in range(cols - divider_col - 1):
                candidates = [color for color, count in colors.items() if i % count == 0]
                if len(candidates) == 1:
                    result[r][divider_col + 1 + i] = candidates[0]
                elif len(candidates) > 1:
                    result[r][divider_col + 1 + i] = max(candidates, key=lambda c: max_pos[c])
    
    elif right_has and not left_has:
        for r in range(rows):
            if grid[r][divider_col] == 0:
                continue
            pattern = [grid[r][c] for c in range(divider_col + 1, cols)]
            colors = {}
            min_pos = {}
            for c_idx, v in enumerate(pattern):
                if v != 0:
                    colors[v] = colors.get(v, 0) + 1
                    if v not in min_pos:
                        min_pos[v] = c_idx
            if not colors:
                continue
            for i in range(divider_col):
                candidates = [color for color, count in colors.items() if i % count == 0]
                if len(candidates) == 1:
                    result[r][divider_col - 1 - i] = candidates[0]
                elif len(candidates) > 1:
                    result[r][divider_col - 1 - i] = min(candidates, key=lambda c: min_pos[c])
    
    return result
