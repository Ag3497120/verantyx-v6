
def transform(grid):
    import numpy as np
    g = np.array(grid)
    h, w = g.shape
    out = g.copy()
    
    # Find reference sequences: consecutive groups of 2+ non-zero in a row
    seqs = []  # list of (vals_list) 
    for r in range(h):
        row = g[r]
        non_zero = [(c, int(row[c])) for c in range(w) if row[c] != 0]
        if len(non_zero) >= 2:
            # Group into consecutive segments
            i = 0
            while i < len(non_zero):
                j = i
                while j + 1 < len(non_zero) and non_zero[j+1][0] == non_zero[j][0] + 1:
                    j += 1
                if j - i >= 1:  # 2+ consecutive
                    seqs.append([val for c, val in non_zero[i:j+1]])
                i = j + 1
    
    # Build lookup: value -> sequence
    val_to_seq = {}
    for seq in seqs:
        for v in seq:
            val_to_seq[v] = seq
    
    # For each row with single isolated non-zero cells, place sequences
    for r in range(h):
        row = g[r]
        non_zero = [(c, int(row[c])) for c in range(w) if row[c] != 0]
        if not non_zero:
            continue
        # Check if this row has isolated single cells (not consecutive)
        for k, (c, v) in enumerate(non_zero):
            # Check if this is isolated (not part of a multi-cell group)
            neighbors = [(c2,v2) for c2,v2 in non_zero if abs(c2-c) == 1]
            if v in val_to_seq:
                seq = val_to_seq[v]
                idx = seq.index(v)
                for i, sv in enumerate(seq):
                    nc = c - idx + i
                    if 0 <= nc < w:
                        out[r, nc] = sv
    
    return out.tolist()
