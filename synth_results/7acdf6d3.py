def transform(grid):
    return _solve(grid)

def solve_7acdf6d3(grid):
    g = [list(r) for r in grid]
    H, W = len(g), len(g[0])
    out = [row[:] for row in g]
    
    # Find 2-shapes (V-shapes: 3 cells with one "gap" in the middle row)
    # Pattern: [2,_,2] in one row, [_,2,_] in next row = triangle
    # Or: [_,2,_] top, [2,_,2] bottom = another triangle
    
    # Remove all 9s from output first
    non_two_non_bg = None
    for r in range(H):
        for c in range(W):
            if g[r][c] not in [0,2,7]:  # assuming bg is 7
                if g[r][c] not in [2]:
                    non_two_non_bg = g[r][c]
    
    # Remove original 9s
    bg = Counter(g[r][c] for r in range(H) for c in range(W)).most_common(1)[0][0]
    for r in range(H):
        for c in range(W):
            if g[r][c] not in [0, bg, 2]:
                out[r][c] = bg
    
    # Find V-shapes of 2s and fill interior with 9
    # Pattern A: two 2s in row r, one 2 in row r+1 between them
    # Pattern B: one 2 in row r, two 2s in row r+1 on either side
    twos = {(r,c) for r in range(H) for c in range(W) if g[r][c]==2}
    
    # Find all 3-cell L-shapes that form a "bowl" or "inverted bowl"
    # Type 1: (r,c1), (r,c2), (r+1,(c1+c2)//2) where c2=c1+2
    for r in range(H-1):
        for c in range(W-2):
            if (r,c) in twos and (r,c+2) in twos and (r+1,c+1) in twos:
                # Bowl: fill interior = (r, c+1)
                out[r][c+1] = 9
    # Type 2: inverted bowl
    for r in range(1, H):
        for c in range(W-2):
            if (r,c) in twos and (r,c+2) in twos and (r-1,c+1) in twos:
                out[r][c+1] = 9
    
    # Also handle larger V-shapes (from 5-cell pattern)
    # Pattern: top row [2,0,0,2], middle rows [2,_,_,2], bottom [0,2,_,2,0]
    # Just flood fill? No - find enclosed regions within 2-boundaries
    
    return out


_solve = solve_7acdf6d3
