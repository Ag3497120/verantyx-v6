def transform(grid):
    from collections import Counter, defaultdict
    R, C = len(grid), len(grid[0])
    bg = Counter(grid[r][c] for r in range(R) for c in range(C)).most_common(1)[0][0]
    ones = [(r,c) for r in range(R) for c in range(C) if grid[r][c]==1]
    sixes = [(r,c) for r in range(R) for c in range(C) if grid[r][c]==6]
    out = [row[:] for row in grid]
    
    # Group 1s by main diagonal (r-c) and anti-diagonal (r+c)
    main_groups = defaultdict(list)
    anti_groups = defaultdict(list)
    for r,c in ones:
        main_groups[r-c].append((r,c))
        anti_groups[r+c].append((r,c))
    
    # Find paired groups (2+ elements)
    paired_main = {k: sorted(v) for k,v in main_groups.items() if len(v)>=2}
    paired_anti = {k: sorted(v) for k,v in anti_groups.items() if len(v)>=2}
    
    # Track which diagonal type each 1-pair uses
    pair_diags = {}  # (type, key) pairs
    
    # Draw 1-trails for main diagonal pairs
    for key, pts in paired_main.items():
        r0,c0 = pts[0]; r1,c1 = pts[-1]
        r,c = r0,c0
        while True:
            if 0<=r<R and 0<=c<C:
                if grid[r][c]==6:
                    out[r][c]=6
                else:
                    out[r][c]=1
            if r==r1 and c==c1: break
            r += (1 if r1>r0 else -1); c += (1 if c1>c0 else -1)
        pair_diags[('main', key)] = True
    
    # Draw 1-trails for anti-diagonal pairs
    for key, pts in paired_anti.items():
        r0,c0 = pts[0]; r1,c1 = pts[-1]
        dc = 1 if c1>c0 else -1
        r,c = r0,c0
        while True:
            if 0<=r<R and 0<=c<C:
                if grid[r][c]==6:
                    out[r][c]=6
                else:
                    out[r][c]=1
            if r==r1 and c==c1: break
            r+=1; c+=dc
        pair_diags[('anti', key)] = True
    
    # 6s that lie on a 1-pair diagonal emit perpendicular rays
    for sr,sc in sixes:
        emitted = False
        # Check if on a paired main diagonal
        if ('main', sr-sc) in pair_diags:
            # emit anti-diagonal ray
            pk = sr+sc
            for row in range(R):
                col = pk-row
                if 0<=col<C:
                    if out[row][col] not in (1,):
                        out[row][col]=6
            emitted = True
        if ('anti', sr+sc) in pair_diags:
            # emit main diagonal ray
            pk = sr-sc
            for row in range(R):
                col = row-pk
                if 0<=col<C:
                    if out[row][col] not in (1,):
                        out[row][col]=6
            emitted = True
        # Ensure 6 itself is placed
        out[sr][sc]=6
    
    return out
