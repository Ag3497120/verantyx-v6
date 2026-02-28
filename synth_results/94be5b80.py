def transform(grid):
    import numpy as np
    g = np.array(grid)
    R, C = g.shape
    
    # Find the key: a block of 3 rows where each col has same color repeated 3x
    key_colors = None
    key_r0 = None
    key_c0 = None
    key_width = None
    
    for r in range(R-2):
        for c in range(C-1):
            if g[r,c] != 0 and g[r+1,c] != 0 and g[r+2,c] != 0 and g[r,c]==g[r+1,c]==g[r+2,c]:
                # Find full width of key block
                w = 1
                while c+w < C and g[r,c+w] != 0 and g[r,c+w]==g[r+1,c+w]==g[r+2,c+w]:
                    w += 1
                if w >= 2:
                    key_colors = g[r, c:c+w].tolist()
                    key_r0, key_c0, key_width = r, c, w
                    break
        if key_colors: break
    
    if not key_colors:
        return grid
    
    # Find existing shape bands per color
    shape_bands = {}  # color -> (r_start, r_end, pattern_2d)
    for color in set(key_colors):
        rows_with_color = [r for r in range(R) if np.any(g[r,:]==color) and not (key_r0<=r<key_r0+3 and any(g[r,key_c0:key_c0+key_width]==color))]
        if rows_with_color:
            r1, r2 = min(rows_with_color), max(rows_with_color)
            # Pattern = which cells have color
            sub = g[r1:r2+1, :]
            pattern = (sub == color)
            shape_bands[color] = (r1, r2, pattern)
    
    # Get template (shape mask, any existing color)
    # All shapes should have same mask
    existing = [(c, r1, r2, pat) for c,(r1,r2,pat) in shape_bands.items()]
    if not existing:
        return grid
    existing.sort(key=lambda x: x[1])
    
    band_h = existing[0][2] - existing[0][1] + 1
    template_mask = existing[0][3]
    
    # Determine order from key and fill in missing
    out = g.copy()
    # Remove key
    out[key_r0:key_r0+3, key_c0:key_c0+key_width] = 0
    
    # For each existing band, map key position
    key_pos = {c: key_colors.index(c) for c in shape_bands}
    
    # Find reference: first existing band (lowest key position)
    ref_color = min(shape_bands.keys(), key=lambda c: key_pos[c])
    ref_ki = key_pos[ref_color]
    ref_r1 = shape_bands[ref_color][0]
    
    for ki, c in enumerate(key_colors):
        if c not in shape_bands:
            offset = ki - ref_ki
            new_r1 = ref_r1 + offset * band_h
            if 0 <= new_r1 < R and 0 <= new_r1 + band_h - 1 < R:
                for dr in range(band_h):
                    for dc in range(C):
                        if template_mask[dr, dc]:
                            out[new_r1+dr, dc] = c
    
    return out.tolist()
