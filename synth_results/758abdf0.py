def transform(grid):
    return _solve(grid)

def solve_758abdf0(grid):
    g = [list(r) for r in grid]
    H, W = len(g), len(g[0])
    out = [row[:] for row in g]
    
    # Find 0-border: is it column 0 or row 0?
    col0_all_zero = all(g[r][0] == 0 for r in range(H))
    row0_all_zero = all(g[0][c] == 0 for c in range(W))
    
    if col0_all_zero:
        # Vertical orientation: 8s in rows, 0-border at col 0
        # For each row: find 8s in that row (excluding col 0)
        for r in range(H):
            row = g[r][1:]  # exclude col 0
            eights_cols = [c+1 for c, v in enumerate(row) if v == 8]
            if not eights_cols: continue
            # Count consecutive groups from left
            # Find the leftmost group of adjacent 8s
            run_start = eights_cols[0]
            run_end = run_start
            for ec in eights_cols[1:]:
                if ec == run_end + 1:
                    run_end = ec
                else:
                    break
            run_len = run_end - run_start + 1
            
            if run_len >= 2:
                # Double 8s -> remove and place 0,0 at end
                for c in range(run_start, run_end+1):
                    out[r][c] = 7 if g[r][c] == 8 else g[r][c]
                for c in range(run_start, run_end+1):
                    out[r][c] = 7
                # Actually find background value
                bg = g[r][2] if g[r][2] not in [0,8] else 7
                # Set run positions to bg
                for c in range(run_start, run_end+1):
                    out[r][c] = bg
                # Place 0,0 at end (last run_len cols)
                for i in range(run_len):
                    tc = W - 1 - i
                    if tc > 0:
                        out[r][tc] = 0
            else:
                # Single 8 -> extend to double
                if eights_cols[0]+1 < W and g[r][eights_cols[0]+1] != 8:
                    out[r][eights_cols[0]+1] = 8
    else:
        # Horizontal orientation: 8s in cols, 0-border at row 0
        for c in range(W):
            col = [g[r][c] for r in range(1, H)]
            eights_rows = [r+1 for r, v in enumerate(col) if v == 8]
            if not eights_rows: continue
            run_start = eights_rows[0]
            run_end = run_start
            for er in eights_rows[1:]:
                if er == run_end + 1:
                    run_end = er
                else:
                    break
            run_len = run_end - run_start + 1
            if run_len >= 2:
                bg = 7  # find bg
                for r in range(run_start, run_end+1):
                    out[r][c] = bg
                for i in range(run_len):
                    tr = H - 1 - i
                    if tr > 0:
                        out[tr][c] = 0
            else:
                if eights_rows[0]+1 < H and g[eights_rows[0]+1][c] != 8:
                    out[eights_rows[0]+1][c] = 8
    return out


_solve = solve_758abdf0
