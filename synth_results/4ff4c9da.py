
def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    # Find row separators (all-zero rows) and col separators
    row_seps = [i for i in range(rows) if all(g[i,:]==0)]
    col_seps = [j for j in range(cols) if all(g[:,j]==0)]
    
    # Find the special color (not 0, not the majority)
    vals = g.flatten()
    cnt = Counter(vals.tolist())
    del cnt[0]
    if not cnt:
        return grid
    bg_color = cnt.most_common(1)[0][0]
    special_colors = [c for c in cnt if c != bg_color]
    if not special_colors:
        return grid
    
    # Find position of special cell
    special_color = special_colors[0]
    pos = list(zip(*np.where(g == special_color)))
    if not pos:
        return grid
    sr, sc = pos[0]
    
    # Find which row-block and col-block contains the special cell
    def get_block(seps, idx, total):
        blocks = []
        prev = -1
        for s in seps:
            blocks.append((prev+1, s-1))
            prev = s
        blocks.append((prev+1, total-1))
        for i,(a,b) in enumerate(blocks):
            if a <= idx <= b:
                return i, a, b
        return -1, -1, -1
    
    rb, ra, rb_end = get_block(row_seps, sr, rows)
    cb, ca, cb_end = get_block(col_seps, sc, cols)
    
    result = g.copy()
    # Propagate special color along entire row-band
    for r in range(ra, rb_end+1):
        for c in range(cols):
            if g[r,c] == bg_color:
                result[r,c] = special_color
    # Propagate along entire col-band
    for r in range(rows):
        for c in range(ca, cb_end+1):
            if g[r,c] == bg_color:
                result[r,c] = special_color
    return result.tolist()
