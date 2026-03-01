def transform(grid):
    import numpy as np
    from collections import Counter
    
    g = np.array(grid)
    rows, cols = g.shape
    
    div_rows = sorted([r for r in range(rows) if (g[r]==5).sum() > cols//3])
    div_cols = sorted([c for c in range(cols) if (g[:,c]==5).sum() > rows//3])
    
    b_rows = [-1] + div_rows + [rows]
    b_cols = [-1] + div_cols + [cols]
    
    nb_r = len(b_rows)-1
    nb_c = len(b_cols)-1
    
    # Block sizes (filter out zero-size blocks)
    block_heights = [b_rows[i+1]-b_rows[i]-1 for i in range(nb_r) if b_rows[i+1]-b_rows[i]-1 > 0]
    block_widths = [b_cols[j+1]-b_cols[j]-1 for j in range(nb_c) if b_cols[j+1]-b_cols[j]-1 > 0]
    bsize_r = Counter(block_heights).most_common(1)[0][0]
    bsize_c = Counter(block_widths).most_common(1)[0][0]
    
    def get_block(bi, bj):
        r1 = b_rows[bi]+1
        c1 = b_cols[bj]+1
        if b_rows[bi+1]-b_rows[bi]-1 < 1 or b_cols[bj+1]-b_cols[bj]-1 < 1:
            return np.zeros((bsize_r, bsize_c), dtype=int)
        return g[r1:r1+bsize_r, c1:c1+bsize_c]
    
    bg = 0
    
    # Top blocks (bi=0): each has one dominant color and pattern
    pattern_blocks = {}
    block_colors = {}
    
    for bj in range(nb_c):
        blk = get_block(0, bj)
        vals = [v for v in blk.flatten() if v != bg]
        if vals:
            color = Counter(vals).most_common(1)[0][0]
            if color != 5:
                block_colors[bj] = color
                pattern_blocks[color] = (blk == color)
    
    # Key blocks (bi=nb_r-1): find the key value for each column
    key_values = {}
    for bj in range(nb_c):
        blk = get_block(nb_r-1, bj)
        vals = [v for v in blk.flatten() if v != bg and v != 5]
        if vals:
            key_values[bj] = Counter(vals).most_common(1)[0][0]
    
    # Build mapping: color -> key_value
    color_to_key = {}
    for bj in range(nb_c):
        if bj in block_colors and bj in key_values:
            color_to_key[block_colors[bj]] = key_values[bj]
    
    # Middle blocks (bi=1): layout
    layout = np.zeros((bsize_r, bsize_c), dtype=int)
    for bj in range(nb_c):
        blk = get_block(1, bj)
        for r in range(bsize_r):
            for c in range(bsize_c):
                v = blk[r, c]
                if v != bg and v != 5:
                    layout[r, c] = v
    
    # Build output
    out = np.zeros((bsize_r * bsize_r, bsize_c * bsize_c), dtype=int)
    
    for r in range(bsize_r):
        for c in range(bsize_c):
            color = layout[r, c]
            if color == bg:
                continue
            key = color_to_key.get(color, color)
            pat = pattern_blocks.get(color, None)
            if pat is None:
                continue
            r1, r2 = r * bsize_r, (r+1) * bsize_r
            c1, c2 = c * bsize_c, (c+1) * bsize_c
            out[r1:r2, c1:c2] = pat * key
    
    return out.tolist()
