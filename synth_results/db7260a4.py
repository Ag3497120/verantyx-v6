
def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]
    one_r, one_c = -1, -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                one_r, one_c = r, c
                out[r][c] = 0
    if one_r == -1:
        return out
    from collections import Counter
    col_counts = Counter()
    twos = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                twos.add((r, c))
                col_counts[c] += 1
    if not twos:
        return out
    vertical_cols = set(c for c, cnt in col_counts.items() if cnt > 1)
    left_walls = sorted([c for c in vertical_cols if c < one_c], reverse=True)
    right_walls = sorted([c for c in vertical_cols if c > one_c])
    if left_walls and right_walls:
        left_wall = left_walls[0]
        right_wall = right_walls[0]
        interior_cols = list(range(left_wall + 1, right_wall))
        if not interior_cols:
            for c in range(cols):
                out[rows - 1][c] = 1
            return out
        floor_row = -1
        for r in range(rows):
            if any(grid[r][c] == 2 for c in interior_cols):
                floor_row = r
                break
        if floor_row == -1:
            for c in range(cols):
                out[rows - 1][c] = 1
        else:
            floor_complete = all(grid[floor_row][c] == 2 for c in range(left_wall, right_wall + 1))
            if not floor_complete:
                for c in range(cols):
                    out[rows - 1][c] = 1
            else:
                wall_start = rows
                for r, c in twos:
                    if c == left_wall or c == right_wall:
                        wall_start = min(wall_start, r)
                for r in range(wall_start, floor_row):
                    for c in interior_cols:
                        if grid[r][c] == 0:
                            out[r][c] = 1
    else:
        for c in range(cols):
            out[rows - 1][c] = 1
    return out
