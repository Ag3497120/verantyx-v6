def transform(grid):
    from collections import Counter
    rows = len(grid)
    cols = len(grid[0])
    flat = [v for r in grid for v in r]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find all 3x3 ring patterns
    patterns = []
    for r in range(rows-2):
        for c in range(cols-2):
            block = [[grid[r+dr][c+dc] for dc in range(3)] for dr in range(3)]
            border = [block[dr][dc] for dr in range(3) for dc in range(3) if (dr,dc)!=(1,1)]
            center = block[1][1]
            b_set = set(v for v in border if v!=bg)
            if len(b_set)==1 and center!=bg and center not in b_set:
                patterns.append((b_set.pop(), center))
    
    if not patterns:
        return grid
    
    ring_values = set(r for r,c in patterns)
    
    # Find the pair where center is also a ring value
    for ring_val, center_val in patterns:
        if center_val in ring_values:
            rv, cv = ring_val, center_val
            return [
                [rv, rv, rv],
                [rv, cv, rv],
                [rv, rv, rv]
            ]
    
    # Fallback: last pattern
    rv, cv = patterns[-1]
    return [[rv,rv,rv],[rv,cv,rv],[rv,rv,rv]]
