
def transform(grid):
    import numpy as np
    g = np.array(grid)
    h, w = g.shape
    bg = 0
    
    # Find row groups (consecutive non-empty rows)
    row_has_content = [any(g[r, c] != bg for c in range(w)) for r in range(h)]
    
    groups = []
    i = 0
    while i < h:
        if row_has_content[i]:
            j = i
            while j < h and row_has_content[j]:
                j += 1
            groups.append((i, j-1))
            i = j
        else:
            i += 1
    
    if not groups:
        return g.tolist()
    
    def find_col_clusters(start_r, end_r):
        occupied_cols = set()
        for r in range(start_r, end_r+1):
            for c in range(w):
                if g[r, c] != bg:
                    occupied_cols.add(c)
        sorted_cols = sorted(occupied_cols)
        if not sorted_cols: return []
        clusters = []
        cl_cols = [sorted_cols[0]]
        for i in range(1, len(sorted_cols)):
            if sorted_cols[i] > sorted_cols[i-1] + 1:  # gap >= 2
                clusters.append((min(cl_cols), max(cl_cols)))
                cl_cols = [sorted_cols[i]]
            else:
                cl_cols.append(sorted_cols[i])
        clusters.append((min(cl_cols), max(cl_cols)))
        return clusters
    
    template_group = None
    template_clusters = None
    for g_start, g_end in groups:
        clusters = find_col_clusters(g_start, g_end)
        if len(clusters) > 1:
            template_group = (g_start, g_end)
            template_clusters = clusters
            break
    
    if template_group is None:
        return g.tolist()
    
    base_col_start = template_clusters[0][0]
    other_offsets = [(cl[0] - base_col_start) for cl in template_clusters[1:]]
    
    t_start, t_end = template_group
    symbol_h = t_end - t_start + 1
    base_c0, base_c1 = template_clusters[0]
    symbol_w = base_c1 - base_c0 + 1
    symbol = g[t_start:t_end+1, base_c0:base_c1+1]
    
    out = g.copy()
    
    for g_start, g_end in groups:
        if (g_start, g_end) == template_group:
            continue
        clusters = find_col_clusters(g_start, g_end)
        if not clusters:
            continue
        grp_base_col = clusters[0][0]
        for offset in other_offsets:
            nc = grp_base_col + offset
            for dr in range(symbol_h):
                for dc in range(symbol_w):
                    r, c = g_start + dr, nc + dc
                    if 0 <= r < h and 0 <= c < w:
                        if symbol[dr, dc] != bg:
                            out[r, c] = 8
    
    return out.tolist()
