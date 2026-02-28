def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    
    # Find divider rows (all same non-zero value)
    dividers = [r for r in range(rows) if len(set(g[r])) == 1 and g[r, 0] != 0]
    
    sections = []
    prev = 0
    for d in dividers:
        sections.append((prev, d-1))
        prev = d + 1
    sections.append((prev, rows-1))
    
    # Extract the pattern row from each section
    patterns = []
    for r1, r2 in sections:
        for r in range(r1, r2+1):
            if any(g[r, c] != 0 for c in range(cols)):
                patterns.append(list(g[r]))
                break
        else:
            patterns.append([0] * cols)
    
    def get_nz(pat):
        return [(c, pat[c]) for c in range(len(pat)) if pat[c] != 0]
    
    nz = [get_nz(p) for p in patterns]
    
    # Detect start and end positions of non-zero runs
    starts = [nz_p[0][0] if nz_p else 0 for nz_p in nz]
    ends = [nz_p[-1][0] if nz_p else 0 for nz_p in nz]
    
    d_start = starts[1] - starts[0] if len(starts) > 1 else 0
    d_end = ends[1] - ends[0] if len(ends) > 1 else 0
    
    # Check consistency
    if len(starts) >= 2:
        d_start = starts[-1] - starts[-2]
        d_end = ends[-1] - ends[-2]
    
    next_start = starts[-1] + d_start
    next_end = ends[-1] + d_end
    
    # Build next pattern with same values (from last pattern, shifted)
    last_nz = nz[-1]
    if last_nz:
        # Relative positions and values from last pattern
        last_start = starts[-1]
        shape = [(c - last_start, v) for c, v in last_nz]
        
        next_pat = [0] * cols
        for dc, v in shape:
            nc = next_start + dc
            if dc == len(shape) - 1:
                nc = next_end  # ensure end matches
            if 0 <= nc < cols:
                next_pat[nc] = v
        
        # Fill between next_start and next_end
        if next_start <= next_end:
            # Use the value pattern from the last row but extended/shifted
            # Check if shape grows
            old_len = len(last_nz)
            new_len = next_end - next_start + 1
            if new_len > old_len:
                # Growing pattern: fill with first value
                fill_val = last_nz[0][1]
                next_pat = [0] * cols
                for c in range(next_start, min(next_end+1, cols)):
                    next_pat[c] = fill_val
                # Add any trailing different values
                last_values = [v for _, v in last_nz]
                for i, v in enumerate(last_values):
                    nc = next_end - (old_len - 1 - i)
                    if 0 <= nc < cols:
                        next_pat[nc] = v
            else:
                next_pat = [0] * cols
                for dc, v in shape:
                    nc = next_start + dc
                    if 0 <= nc < cols:
                        next_pat[nc] = v
    else:
        next_pat = [0] * cols
    
    return [[0]*cols, next_pat, [0]*cols]
