def transform(grid):
    return _solve(grid)

def solve_73c3b0d8(grid):
    g = [list(r) for r in grid]
    H, W = len(g), len(g[0])
    # Find divider row (all 2s)
    div_row = -1
    for r in range(H):
        if all(g[r][c] == 2 for c in range(W)):
            div_row = r; break
    if div_row == -1: return grid
    
    out = [row[:] for row in g]
    # Clear all 4s above divider (they will be replaced)
    for r in range(div_row):
        for c in range(W):
            if out[r][c] == 4:
                out[r][c] = 0
    # Also clear 4s below divider
    for r in range(div_row+1, H):
        for c in range(W):
            if out[r][c] == 4:
                out[r][c] = 0
    
    # Process each 4
    original_fours_above = [(r,c) for r in range(div_row) for c in range(W) if g[r][c]==4]
    original_fours_below = [(r,c) for r in range(div_row+1,H) for c in range(W) if g[r][c]==4]
    
    # All 4s move 1 step down
    new_positions = {}
    for r, c in original_fours_above + original_fours_below:
        new_r = r + 1
        if 0 <= new_r < H:
            if (new_r, c) not in new_positions:
                new_positions[(new_r, c)] = 4
    
    # Place 4s and V-shapes
    for (new_r, new_c), val in new_positions.items():
        out[new_r][new_c] = 4
        # If this 4 is at divider-1 (row just above divider)
        if new_r == div_row - 1:
            # Generate V-shape going up from here
            step = 1
            while True:
                left_c = new_c - step
                right_c = new_c + step
                placed = False
                for vc, vr in [(left_c, new_r-step), (right_c, new_r-step)]:
                    if 0 <= vr < div_row and 0 <= vc < W:
                        out[vr][vc] = 4
                        placed = True
                if not placed: break
                step += 1
                if step > max(W, div_row): break
    
    return out


_solve = solve_73c3b0d8
