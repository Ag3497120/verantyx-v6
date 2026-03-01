def transform(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    rows, cols = g.shape
    bg = 8
    out = g.copy()
    
    from collections import defaultdict
    
    # Group marks by intercept (r+c for slope -1 lines)
    intercept_marks = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            if g[r][c] != bg:
                intercept_marks[r+c].append(r-c)
    
    if not intercept_marks:
        return out.tolist()
    
    # Split each intercept into connected segments (consecutive perp positions)
    segments = []  # list of (intercept, perp_min, perp_max)
    for i, perps in intercept_marks.items():
        perps_sorted = sorted(set(perps))
        # Split into consecutive groups (gap > 2 means separate segment)
        seg_start = perps_sorted[0]
        prev = perps_sorted[0]
        for j in range(1, len(perps_sorted)):
            if perps_sorted[j] - prev > 2:
                segments.append((i, seg_start, prev))
                seg_start = perps_sorted[j]
            prev = perps_sorted[j]
        segments.append((i, seg_start, prev))
    
    # For each cell, check if it's between two segments that cover its perp position
    # Sort segments by intercept
    segments.sort()
    
    for r in range(rows):
        for c in range(cols):
            if out[r][c] != bg:
                continue
            s = r + c
            p = r - c
            
            left = None
            right = None
            for intercept, pmin, pmax in segments:
                if pmin <= p <= pmax:
                    if intercept < s:
                        left = intercept
                    elif intercept > s:
                        if right is None:
                            right = intercept
                        break
            
            if left is not None and right is not None:
                out[r][c] = 2
    
    return out.tolist()
