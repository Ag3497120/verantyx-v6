import numpy as np

def transform(grid):
    g = np.array(grid)
    out = g.copy()
    h, w = g.shape
    bg = 0
    
    def contiguous_ranges(arr, val):
        ranges = []
        start = None
        for i, v in enumerate(arr):
            if v == val:
                if start is None: start = i
            else:
                if start is not None and i - start >= 2:
                    ranges.append((start, i-1))
                start = None
        if start is not None and len(arr) - start >= 2:
            ranges.append((start, len(arr)-1))
        return ranges
    
    # Find V-5-lines
    v_lines = []
    for c in range(w):
        for r1, r2 in contiguous_ranges(g[:, c], 5):
            v_lines.append((c, r1, r2))
    
    # Find H-5-lines
    h_lines = []
    for r in range(h):
        for c1, c2 in contiguous_ranges(g[r, :], 5):
            h_lines.append((r, c1, c2))
    
    def count_non_bg_non5(row, exclude_col):
        return sum(1 for col, val in enumerate(row) if col != exclude_col and val != bg and val != 5)
    
    # Process V-5-lines
    for c, r1, r2 in v_lines:
        counts = [(count_non_bg_non5(g[r, :], c), r) for r in range(r1, r2+1)]
        max_cnt = max(x[0] for x in counts)
        if max_cnt == 0:
            continue
        # Find contiguous block of max_cnt rows - try bottom first, then top
        pat_rows = None
        for direction in ['bottom', 'top']:
            rows_iter = range(r2, r1-1, -1) if direction == 'bottom' else range(r1, r2+1)
            block = []
            for r in rows_iter:
                if count_non_bg_non5(g[r, :], c) == max_cnt:
                    block.append(r)
                else:
                    break
            if block:
                pat_rows = sorted(block)
                break
        
        if not pat_rows:
            continue
        
        pat = g[pat_rows[0]:pat_rows[-1]+1, :].copy()
        pat_h = len(pat_rows)
        
        for i in range(r2 - r1 + 1):
            r = r1 + i
            out[r, :] = pat[i % pat_h, :]
            out[r, c] = 5
    
    # Process H-5-lines
    for r, c1, c2 in h_lines:
        def count_col(c):
            return sum(1 for row in range(h) if row != r and out[row, c] != bg and out[row, c] != 5)
        
        counts = [(count_col(c), c) for c in range(c1, c2+1)]
        max_cnt = max(x[0] for x in counts) if counts else 0
        
        if max_cnt == 0:
            # Try from original g
            def count_col_g(c):
                return sum(1 for row in range(h) if row != r and g[row, c] != bg and g[row, c] != 5)
            counts = [(count_col_g(c), c) for c in range(c1, c2+1)]
            max_cnt = max(x[0] for x in counts) if counts else 0
            if max_cnt == 0:
                continue
            src = g
        else:
            src = out
        
        # Find contiguous block of max_cnt cols
        pat_cols = None
        for direction in ['left', 'right']:
            cols_iter = range(c1, c2+1) if direction == 'left' else range(c2, c1-1, -1)
            block = []
            for c in cols_iter:
                cnt = sum(1 for row in range(h) if row != r and src[row, c] != bg and src[row, c] != 5)
                if cnt == max_cnt:
                    block.append(c)
                else:
                    break
            if block:
                pat_cols = sorted(block)
                break
        
        if not pat_cols:
            continue
        
        pat = src[:, pat_cols[0]:pat_cols[-1]+1].copy()
        pat_w = len(pat_cols)
        
        for j in range(c2 - c1 + 1):
            c = c1 + j
            # Only update non-bg rows from pattern
            for row in range(h):
                if row != r:
                    val = pat[row, j % pat_w]
                    if val != bg:
                        out[row, c] = val
                    # if val is bg, keep whatever is already in out
        out[r, c1:c2+1] = 5
    
    return out.tolist()
