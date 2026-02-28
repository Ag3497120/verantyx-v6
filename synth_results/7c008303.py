def transform(grid):
    import numpy as np
    arr = np.array(grid)
    rows, cols = arr.shape
    
    # Find 8-separator row and column
    sep_rows = [i for i in range(rows) if all(arr[i,:]==8)]
    sep_cols = [j for j in range(cols) if all(arr[:,j]==8)]
    
    if not sep_rows or not sep_cols:
        return grid
    
    sr, sc = sep_rows[0], sep_cols[0]
    
    # 4 quadrants
    quads = {
        'TL': arr[:sr, :sc],
        'TR': arr[:sr, sc+1:],
        'BL': arr[sr+1:, :sc],
        'BR': arr[sr+1:, sc+1:],
    }
    
    # Find pattern quad (has 3s) and key quad (has non-0, non-8, non-3 values)
    pattern_quad = None
    key_quad = None
    pattern_name = None
    key_name = None
    
    for name, q in quads.items():
        vals = set(q.flatten()) - {0, 8}
        if 3 in vals:
            pattern_quad = q
            pattern_name = name
        elif vals:
            key_quad = q
            key_name = name
    
    if pattern_quad is None or key_quad is None:
        return grid
    
    ph, pw = pattern_quad.shape
    kh, kw = key_quad.shape
    
    # Block size
    bh = ph // kh
    bw = pw // kw
    
    result = np.zeros_like(pattern_quad)
    for r in range(ph):
        for c in range(pw):
            if pattern_quad[r, c] == 3:
                kr = r // bh
                kc = c // bw
                if kr < kh and kc < kw:
                    result[r, c] = key_quad[kr, kc]
    
    return result.tolist()