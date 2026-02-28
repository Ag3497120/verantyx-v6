def transform(grid):
    import numpy as np
    from collections import Counter
    g = np.array(grid)
    H, W = g.shape
    # Find separator color
    all_colors = []
    for r in range(H):
        vals = np.unique(g[r])
        if len(vals) == 1 and vals[0] != 0:
            all_colors.append(vals[0])
    if not all_colors:
        return grid
    sep_color = Counter(all_colors).most_common(1)[0][0]
    sep_rows = [r for r in range(H) if np.all(g[r] == sep_color)]
    sep_cols = [c for c in range(W) if np.all(g[:, c] == sep_color)]
    
    def make_bands(seps, size):
        b = []; prev = 0
        for s in seps:
            if prev < s: b.append((prev, s-1))
            prev = s + 1
        if prev < size: b.append((prev, size-1))
        return b
    
    rb = make_bands(sep_rows, H)
    cb = make_bands(sep_cols, W)
    
    # Count cells with each color (one count per cell, not per pixel)
    color_counts = Counter()
    for r1, r2 in rb:
        for c1, c2 in cb:
            cell = g[r1:r2+1, c1:c2+1].flatten()
            unique_colors = [v for v in set(cell) if v != 0 and v != sep_color]
            for col in unique_colors:
                color_counts[col] += 1
    
    if not color_counts:
        return [[]]
    
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1])
    max_count = sorted_colors[-1][1]
    out = []
    for color, count in sorted_colors:
        row = [color] * count + [0] * (max_count - count)
        out.append(row)
    return out
