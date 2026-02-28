def transform(grid):
    import numpy as np
    from collections import Counter
    
    g = np.array(grid)
    rows, cols = g.shape
    
    # Find tile structure by detecting "separator zones" (runs of mostly-zero rows/cols)
    # Threshold: positions with >75% zero are considered separators
    threshold = 0.75
    
    col_zero_frac = np.mean(g == 0, axis=0)
    row_zero_frac = np.mean(g == 0, axis=1)
    
    def find_tile_zones(zero_frac, length):
        """Return list of (start, end) for tile zones (non-separator runs)"""
        is_sep = zero_frac > threshold
        zones = []
        in_tile = False
        start = 0
        for i in range(length):
            if not is_sep[i] and not in_tile:
                start = i
                in_tile = True
            elif is_sep[i] and in_tile:
                zones.append((start, i))
                in_tile = False
        if in_tile:
            zones.append((start, length))
        return zones
    
    col_tile_zones = find_tile_zones(col_zero_frac, cols)
    row_tile_zones = find_tile_zones(row_zero_frac, rows)
    
    if not col_tile_zones or not row_tile_zones:
        return grid
    
    # Find the most common tile size
    col_widths = Counter(e-s for s,e in col_tile_zones)
    row_heights = Counter(e-s for s,e in row_tile_zones)
    tile_w = col_widths.most_common(1)[0][0]
    tile_h = row_heights.most_common(1)[0][0]
    
    # Filter to tile-sized zones only
    col_tile_zones = [(s,e) for s,e in col_tile_zones if e-s == tile_w]
    row_tile_zones = [(s,e) for s,e in row_tile_zones if e-s == tile_h]
    
    if not col_tile_zones or not row_tile_zones:
        return grid
    
    # Majority vote for each tile position
    tile = np.zeros((tile_h, tile_w), dtype=int)
    for ir in range(tile_h):
        for ic in range(tile_w):
            vals = []
            for rs, re in row_tile_zones:
                for cs, ce in col_tile_zones:
                    r = rs + ir
                    c = cs + ic
                    if r < rows and c < cols:
                        vals.append(int(g[r, c]))
            if vals:
                cnt = Counter(vals)
                tile[ir, ic] = cnt.most_common(1)[0][0]
    
    # Reconstruct output
    result = np.zeros_like(g)
    for rs, re in row_tile_zones:
        for cs, ce in col_tile_zones:
            result[rs:rs+tile_h, cs:cs+tile_w] = tile
    
    return result.tolist()
