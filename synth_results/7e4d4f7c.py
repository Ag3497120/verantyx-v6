def transform(grid):
    return _solve(grid)

def solve_7e4d4f7c(grid):
    g = grid
    row0 = g[0]
    row1 = g[1]
    
    # Background = most common in rows 1+
    from collections import Counter
    all_vals = [v for row in g[1:] for v in row]
    bg = Counter(all_vals).most_common(1)[0][0]
    
    # Background of row 0
    row0_bg = bg
    
    # col-0 values in rows 1+
    col0_vals = [g[r][0] for r in range(1, len(g)) if g[r][0] != bg]
    col0_marker = col0_vals[0] if col0_vals else 0
    
    # Build row 2
    row2 = []
    for c in range(len(row0)):
        v = row0[c]
        if col0_marker == 6:
            if v == row0_bg:
                row2.append(6)
            else:
                row2.append(v)
        else:
            if v != row0_bg:
                row2.append(6)
            else:
                row2.append(v)
    
    return [list(row0), list(row1), row2]


_solve = solve_7e4d4f7c
