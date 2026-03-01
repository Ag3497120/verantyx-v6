def transform(grid):
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    ones = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    sixes = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 6]
    for r1,c1 in ones:
        six_partner = None
        for r2,c2 in sixes:
            if r2==r1 or c2==c1:
                six_partner = (r2,c2); break
        if not six_partner: continue
        r2,c2 = six_partner
        dr = r2-r1; dc = c2-c1
        dist = abs(dr)+abs(dc)
        dr_n = 0 if dr==0 else dr//abs(dr)
        dc_n = 0 if dc==0 else dc//abs(dc)
        new_dr = -dc_n; new_dc = dr_n
        tr = r1+new_dr*dist; tc = c1+new_dc*dist
        if 0<=tr<rows and 0<=tc<cols and grid[tr][tc]==8:
            result[tr][tc] = 7
    for r,c in sixes:
        result[r][c] = 8
    return result
