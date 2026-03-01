import numpy as np
from itertools import permutations

def transform(grid):
    arr = np.array(grid)
    rows, cols = arr.shape
    bg = 7
    result = np.full_like(arr, bg)
    
    sep_rows = [r for r in range(rows) if (arr[r] == bg).all()]
    bounds = []
    start = 0
    for r in sep_rows + [rows]:
        if r > start: bounds.append((start, r-1))
        start = r + 1
    
    bands = []
    for bstart, bend in bounds:
        band_rows = bend - bstart + 1
        eight_per_row = []  # relative rows
        color_per_row = []
        color_val = None
        for i in range(band_rows):
            r = bstart + i
            n8 = int((arr[r] == 8).sum())
            eight_per_row.append(n8)
            non_bg = [arr[r,c] for c in range(cols) if arr[r,c] != bg and arr[r,c] != 8]
            if non_bg and color_val is None: color_val = non_bg[0]
            color_per_row.append(len(non_bg))
        bands.append({'start':bstart, 'end':bend, 'nrows': band_rows,
                      'eight':eight_per_row, 'color':color_per_row, 'val':color_val})
    
    n = len(bands)
    
    def is_rect(eight_list, color_list):
        if len(eight_list) != len(color_list): return False
        totals = set(a+b for a,b in zip(eight_list, color_list))
        return len(totals) == 1 and totals.pop() > 0
    
    # Find matching
    matched_perm = None
    for perm in permutations(range(n)):
        valid = all(
            is_rect(bands[i]['eight'], bands[perm[i]]['color'])
            for i in range(n) if bands[perm[i]]['val'] is not None
        )
        if valid:
            matched_perm = perm
            break
    
    if matched_perm is None:
        matched_perm = list(range(n))  # fallback: identity
    
    # Build output
    for i, (bstart, bend) in enumerate(bounds):
        j = matched_perm[i]
        color_val = bands[j]['val']
        color_shape = bands[j]['color']
        band_rows = bend - bstart + 1
        
        for ri in range(band_rows):
            r = bstart + ri
            eight_cols = [c for c in range(cols) if arr[r,c] == 8]
            for c in eight_cols: result[r, c] = 8
            
            n_color = color_shape[ri] if ri < len(color_shape) else 0
            if n_color > 0 and color_val is not None:
                rightmost_8 = max(eight_cols) if eight_cols else -1
                for k in range(n_color):
                    nc = rightmost_8 + 1 + k
                    if 0 <= nc < cols: result[r, nc] = color_val
    
    return result.tolist()
