def transform(grid):
    rows, cols = len(grid[0]), None  # find from context
    rows = len(grid)
    cols = len(grid[0])
    
    # Find the 2
    pos = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                pos = (r, c)
    
    if not pos:
        return grid
    
    sr, sc = pos
    out = [list(row) for row in grid]
    
    # Pattern: from row sr, expand diagonally downward (2s arms going outward)
    # Also: below 2, 1s going diagonally further
    # From train[0]: input has 2 at row 0, col 2. Output grows down with 2s diverging and 1s diverging further
    # row 0: [0,0,2,0,0] - 2 at col 2
    # row 1: [0,2,0,2,0] - 2s at col 1,3 (+/-1)
    # row 2: [2,0,0,0,2] - 2s at col 0,4 (+/-2)
    # row 3: [0,1,0,0,0] - 1 at col 1
    # row 4: [0,0,1,0,0] - 1 at col 2
    # So: 2s expand left and right by 1 each row for rows rows of total expansion
    # Then 1s: below that, 1s go diagonally inward toward center
    
    # Actually let me look more carefully:
    # The 2s form an inverted V (going downward-outward)
    # When they hit edges, they turn into 1s going in opposite (inward-down) direction
    
    # Let's simulate: start at (sr, sc), expand outward
    # At each row r below sr: left arm at sc-(r-sr), right arm at sc+(r-sr)
    # When an arm goes out of bounds, mark 1s going the other way
    
    for r in range(sr, rows):
        d = r - sr
        lc = sc - d
        rc = sc + d
        if 0 <= lc < cols:
            out[r][lc] = 2
        if 0 <= rc < cols and rc != lc:
            out[r][rc] = 2
        # if out of bounds, mark 1 continuing
    
    # Add 1s where arms hit boundary
    # Check train[0] more carefully:
    # row 3: [0,1,0,0,0] → 1 at col 1. d=3: lc=-1 (OOB), rc=5 (OOB). But these are 1s...
    # Hmm: 1 at col 1, not col -1 or 5. 
    # Actually maybe: when arm goes OOB, the remaining "energy" reflects back?
    # lc at row 3 = 2-3=-1, which bounces to become col 1 (2*0-(-1)=1)?
    # That would be 2*0 + 1 = 1 from left edge? Let me think:
    # Left edge = col 0. At row 3, left arm would be at col -1. Reflected off left edge: new_col = -lc = 1.
    # So out[3][1] = 1. ✓
    # At row 4: left arm = -2, reflected = 2. But right arm = 6 (OOB), reflected from right edge col 4: new_col = 2*4 - 6 = 2. So both reflections land at col 2.
    # out[4][2] = 1. ✓
    
    # Let me fix this:
    out = [list(row) for row in grid]
    out[sr][sc] = 2  # keep original
    
    for r in range(sr+1, rows):
        d = r - sr
        for sign in [-1, 1]:
            c = sc + sign * d
            if 0 <= c < cols:
                out[r][c] = 2
            else:
                # reflect
                if c < 0:
                    rc = -c
                else:
                    rc = 2*(cols-1) - c
                if 0 <= rc < cols:
                    out[r][rc] = 1
    
    return out
