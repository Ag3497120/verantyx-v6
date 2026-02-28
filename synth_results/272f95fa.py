def transform(grid):
    import numpy as np
    arr = np.array(grid)
    h, w = arr.shape
    
    # Find all 8 cells
    eight_pos = np.argwhere(arr == 8)
    
    # Determine if we have vertical or horizontal lines of 8s
    # Check for vertical lines: same column, consecutive rows
    vertical_lines = []
    for c in range(w):
        col = arr[:, c]
        runs = []
        start = None
        for r in range(h):
            if col[r] == 8:
                if start is None:
                    start = r
            else:
                if start is not None:
                    runs.append((start, r-1))
                    start = None
        if start is not None:
            runs.append((start, h-1))
        for s, e in runs:
            if e - s >= 1:  # at least 2 consecutive
                vertical_lines.append((c, s, e))
    
    # Check for horizontal lines: same row, consecutive columns
    horizontal_lines = []
    for r in range(h):
        row = arr[r, :]
        runs = []
        start = None
        for c in range(w):
            if row[c] == 8:
                if start is None:
                    start = c
            else:
                if start is not None:
                    runs.append((start, c-1))
                    start = None
        if start is not None:
            runs.append((start, w-1))
        for s, e in runs:
            if e - s >= 1:
                horizontal_lines.append((r, s, e))
    
    # Determine intersection points (where vertical and horizontal lines cross)
    intersections = []
    for vc, vr_start, vr_end in vertical_lines:
        for hr, hc_start, hc_end in horizontal_lines:
            if vr_start <= hr <= vr_end and hc_start <= vc <= hc_end:
                intersections.append((hr, vc))
    
    # For each intersection, we need to fill four quadrants
    for ir, ic in intersections:
        # Quadrant 1: above and right of intersection
        for r in range(ir):
            for c in range(ic+1, w):
                if arr[r, c] == 0:
                    arr[r, c] = 2
        # Quadrant 2: below and right of intersection
        for r in range(ir+1, h):
            for c in range(ic+1, w):
                if arr[r, c] == 0:
                    arr[r, c] = 6
        # Quadrant 3: below and left of intersection
        for r in range(ir+1, h):
            for c in range(ic):
                if arr[r, c] == 0:
                    arr[r, c] = 4
        # Quadrant 4: above and left of intersection
        for r in range(ir):
            for c in range(ic):
                if arr[r, c] == 0:
                    arr[r, c] = 1
    
    return arr.tolist()