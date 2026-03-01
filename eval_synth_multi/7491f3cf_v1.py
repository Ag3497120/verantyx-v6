def transform(grid):
    import numpy as np
    from collections import Counter
    g = np.array(grid, dtype=int)
    rows, cols = g.shape
    bg = g[0][0]
    
    sep_cols = set(c for c in range(cols) if all(g[r][c] == bg for r in range(rows)))
    panel_ranges = []
    c = 0
    while c < cols:
        if c not in sep_cols:
            s = c
            while c < cols and c not in sep_cols: c += 1
            panel_ranges.append((s, c))
        else: c += 1
    
    if len(panel_ranges) < 4:
        return grid
    
    inner_rows = [r for r in range(rows) if not all(g[r][c] == bg for c in range(cols))]
    h = len(inner_rows)
    w = panel_ranges[0][1] - panel_ranges[0][0]
    
    def get_panel(idx):
        s, e = panel_ranges[idx]
        return np.array([[g[r][c] for c in range(s, e)] for r in inner_rows])
    
    P1, P2, P3 = get_panel(0), get_panel(1), get_panel(2)
    panel_bg = Counter(int(x) for x in get_panel(3).flatten()).most_common(1)[0][0]
    
    p1_marks = [(r, c) for r in range(h) for c in range(w) if P1[r][c] != panel_bg]
    if not p1_marks:
        return grid
    
    param_fns = [('r', lambda r,c: r), ('c', lambda r,c: c), ('r+c', lambda r,c: r+c), ('r-c', lambda r,c: r-c)]
    
    # Find best parameters
    param_info = {}
    for pname, fn in param_fns:
        counter = Counter(fn(r,c) for r,c in p1_marks)
        val, cnt = counter.most_common(1)[0]
        param_info[pname] = (val, cnt, fn)
    
    max_count = max(v[1] for v in param_info.values())
    best_params = [(pname, v[0], v[2]) for pname, v in param_info.items() if v[1] == max_count]
    
    # For each best parameter, determine P2 side direction
    conditions = []
    for pname, best_val, fn in best_params:
        extras = [(r,c) for r,c in p1_marks if fn(r,c) != best_val]
        if extras:
            extra_val = fn(extras[0][0], extras[0][1])
            p2_leq = extra_val <= best_val
        else:
            p2_leq = True
        conditions.append((fn, best_val, p2_leq))
    
    def is_p2_side(r, c):
        for fn, val, p2_leq in conditions:
            pv = fn(r, c)
            if p2_leq:
                if pv > val: return False
            else:
                if pv < val: return False
        return True
    
    def is_on_boundary(r, c):
        for fn, val, _ in conditions:
            if fn(r, c) == val:
                return True
        return False
    
    def is_extra(r, c):
        return (r, c) in p1_marks and not is_on_boundary(r, c)
    
    out = g.copy()
    p4_start = panel_ranges[3][0]
    
    for ri in range(h):
        for ci in range(w):
            r_out = inner_rows[ri]
            c_out = p4_start + ci
            p2v = int(P2[ri][ci])
            p3v = int(P3[ri][ci])
            
            if is_extra(ri, ci):
                if p2v != panel_bg:
                    out[r_out][c_out] = p2v
            elif is_on_boundary(ri, ci):
                if p2v != panel_bg:
                    out[r_out][c_out] = p2v
                elif p3v != panel_bg:
                    out[r_out][c_out] = p3v
            elif is_p2_side(ri, ci):
                if p2v != panel_bg:
                    out[r_out][c_out] = p2v
            else:
                if p3v != panel_bg:
                    out[r_out][c_out] = p3v
    
    return out.tolist()
