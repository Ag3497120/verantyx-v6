def transform(grid):
    h = len(grid)
    w = len(grid[0])
    # Convert to mutable list of lists
    out = [row[:] for row in grid]
    
    # First, turn all 2s into 7s in output
    for r in range(h):
        for c in range(w):
            if out[r][c] == 2:
                out[r][c] = 7
    
    # Find all 5s and 0s
    fives = []
    zeros = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 5:
                fives.append((r, c))
            elif grid[r][c] == 0:
                zeros.append((r, c))
    
    # For each 5, find matching 0 in same row/col with clear path
    for r5, c5 in fives:
        for r0, c0 in zeros:
            if r5 == r0:  # same row
                start_col = min(c5, c0)
                end_col = max(c5, c0)
                clear = True
                for cc in range(start_col + 1, end_col):
                    if grid[r5][cc] not in (2, 7):
                        clear = False
                        break
                if clear:
                    # Draw horizontal line
                    for cc in range(start_col + 1, end_col):
                        out[r5][cc] = 2
                    # Mark this zero as used (optional, but fine for given examples)
                    break
            elif c5 == c0:  # same column
                start_row = min(r5, r0)
                end_row = max(r5, r0)
                clear = True
                for rr in range(start_row + 1, end_row):
                    if grid[rr][c5] not in (2, 7):
                        clear = False
                        break
                if clear:
                    # Draw vertical line
                    for rr in range(start_row + 1, end_row):
                        out[rr][c5] = 2
                    break
    
    return out