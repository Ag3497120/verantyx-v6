def transform(grid):
    import copy
    rows, cols = len(grid), len(grid[0])
    from collections import Counter
    # Find background
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find non-background cells
    color_cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] != bg]
    
    # Group by anti-diagonal (r+c)
    from collections import defaultdict
    adiag = defaultdict(list)
    for r,c in color_cells:
        adiag[r+c].append((r,c))
    
    # Sort each anti-diagonal group and find consecutive segments
    # (cells that form a contiguous anti-diagonal run)
    # Two cells on same anti-diagonal are adjacent if |r1-r2|==1
    segments = []
    for key in sorted(adiag.keys()):
        cells = sorted(adiag[key])
        # Check if all cells are consecutive (each adjacent to next)
        # A run is consecutive if cells differ by 1 in r
        if len(cells) >= 1:
            rvals = [r for r,c in cells]
            # All consecutive?
            is_consec = all(rvals[i+1]-rvals[i]==1 for i in range(len(rvals)-1))
            if is_consec:
                segments.append((key, cells))
    
    # Sort segments by anti-diag value
    segments.sort(key=lambda x: x[0])
    
    out = copy.deepcopy(grid)
    
    # For consecutive pairs of segments, fill the enclosed region
    for i in range(len(segments)-1):
        a_key, a_cells = segments[i]
        b_key, b_cells = segments[i+1]
        
        # r-c range for A
        a_rc = [r-c for r,c in a_cells]
        a_rc_min, a_rc_max = min(a_rc), max(a_rc)
        
        # r-c range for B
        b_rc = [r-c for r,c in b_cells]
        b_rc_min, b_rc_max = min(b_rc), max(b_rc)
        
        # Intersection
        rc_min = max(a_rc_min, b_rc_min)
        rc_max = min(a_rc_max, b_rc_max)
        
        if rc_min > rc_max:
            continue
        
        # Fill cells: a_key < r+c < b_key, rc_min <= r-c <= rc_max
        for r in range(rows):
            for c in range(cols):
                s = r+c
                d = r-c
                if a_key < s < b_key and rc_min <= d <= rc_max:
                    if grid[r][c] == bg:
                        out[r][c] = 2
    
    return out