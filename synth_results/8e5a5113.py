def transform(grid):
    import numpy as np
    from collections import Counter
    g = np.array(grid)
    H, W = g.shape
    out = g.copy()
    
    # Find separator columns: interior cols where all cells are same non-zero value
    # AND that value appears in multiple such columns (it's the separator)
    sep_candidates = {}
    for c in range(1, W-1):  # interior only
        col = g[:, c]
        u = np.unique(col)
        if len(u) == 1 and u[0] != 0:
            v = int(u[0])
            if v not in sep_candidates: sep_candidates[v] = []
            sep_candidates[v].append(c)
    
    if not sep_candidates:
        return grid
    # Pick the separator value with most occurrences
    sep_val = max(sep_candidates, key=lambda v: len(sep_candidates[v]))
    sep_cols = sorted(sep_candidates[sep_val])
    
    def col_bands(seps, size):
        b = []; prev = 0
        for s in seps:
            if prev < s: b.append((prev, s-1))
            prev = s + 1
        if prev < size: b.append((prev, size-1))
        return b
    
    cbands = col_bands(sep_cols, W)
    if len(cbands) < 2: return grid
    
    # Find source section (non-zero, non-sep data)
    source_idx = None; source = None
    for i, (c1, c2) in enumerate(cbands):
        sec = g[:, c1:c2+1]
        if np.any((sec != 0) & (sec != sep_val)):
            source_idx = i; source = sec.copy(); break
    if source is None: return grid
    
    for i, (c1, c2) in enumerate(cbands):
        if i == source_idx: continue
        offset = i - source_idx
        k = (-offset) % 4
        rotated = np.rot90(source, k=k)
        if g[:, c1:c2+1].shape == rotated.shape:
            out[:, c1:c2+1] = np.where(g[:, c1:c2+1] == 0, rotated, g[:, c1:c2+1])
    return out.tolist()
