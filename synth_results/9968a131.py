def transform(grid):
    import numpy as np
    g = np.array(grid)
    R, C = g.shape
    
    # Find background (most common value)
    bg = int(np.bincount(g.flatten()).argmax())
    
    out = g.copy()
    
    # Find the two alternating row types (by checking if rows alternate)
    # Row type 0 (even indices): unchanged
    # Row type 1 (odd indices): shift non-bg values right by 1
    
    for r in range(1, R, 2):
        row = g[r].copy()
        nz_cols = [c for c in range(C) if row[c] != bg]
        if not nz_cols: continue
        nz_vals = [row[c] for c in nz_cols]
        
        # Clear old positions
        for c in nz_cols:
            out[r, c] = bg
        
        # Shift right by 1 (in column space)
        for i, c in enumerate(nz_cols):
            new_c = c + 1
            if new_c < C:
                out[r, new_c] = nz_vals[i]
    
    return out.tolist()
