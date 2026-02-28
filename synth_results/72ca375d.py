def transform(grid):
    return _solve(grid)

def solve_72ca375d(grid):
    g = np.array(grid)
    colors = set(g.flatten()) - {0}
    best = None
    best_color = None
    for color in colors:
        positions = list(zip(*np.where(g==color)))
        if not positions: continue
        rows = [r for r,c in positions]
        cols = [c for r,c in positions]
        r0, r1, c0, c1 = min(rows), max(rows)+1, min(cols), max(cols)+1
        # Extract bbox region
        region = g[r0:r1, c0:c1]
        # Check LR symmetry (each row is palindrome)
        lr_sym = all((row == row[::-1]).all() for row in region)
        if lr_sym:
            best = (r0, r1, c0, c1)
            best_color = color
            break  # take first LR-symmetric one
    
    if best:
        r0, r1, c0, c1 = best
        # Return bbox region with only the matching color
        region = g[r0:r1, c0:c1].tolist()
        return region
    # Fallback: return most filled region
    return grid


_solve = solve_72ca375d
