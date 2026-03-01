
def transform(grid):
    from collections import Counter
    R, C = len(grid), len(grid[0])
    cnt = Counter(grid[r][c] for r in range(R) for c in range(C))
    bg = cnt.most_common(1)[0][0]
    # Find the cross center
    nz = [(r,c) for r in range(R) for c in range(C) if grid[r][c] != bg]
    if not nz: return [row[:] for row in grid]
    fg = grid[nz[0][0]][nz[0][1]]
    # Identify rows and cols with the cross arms
    row_counts = Counter(r for r,c in nz)
    col_counts = Counter(c for r,c in nz)
    cr = max(row_counts, key=row_counts.get)
    cc = max(col_counts, key=col_counts.get)
    # Find arm extents
    arm_up = cr - min(r for r,c in nz if c==cc)
    arm_down = max(r for r,c in nz if c==cc) - cr
    arm_left = cc - min(c for r,c in nz if r==cr)
    arm_right = max(c for r,c in nz if r==cr) - cc
    out = [row[:] for row in grid]
    # Draw expanding rectangles
    for level in range(1, max(R, C)):
        r0 = cr - arm_up - level
        r1 = cr + arm_down + level
        c0 = cc - arm_left - level
        c1 = cc + arm_right + level
        # Check if entirely out of bounds
        if r0 >= R and r1 < 0 and c0 >= C and c1 < 0: break
        in_bounds = False
        for r in range(max(0,r0), min(R,r1+1)):
            for c in range(max(0,c0), min(C,c1+1)):
                if r==r0 or r==r1 or c==c0 or c==c1:
                    out[r][c] = fg
                    in_bounds = True
        if not in_bounds and level > max(R, C)//2: break
    return out
