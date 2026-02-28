def transform(grid):
    import numpy as np
    g = np.array(grid)
    R, C = g.shape
    vals, cnts = np.unique(g, return_counts=True)
    nz = [(v,c) for v,c in zip(vals,cnts) if v!=0]
    wall, _ = max(nz, key=lambda x: x[1])
    pat_colors = [v for v,c in nz if v!=wall]
    if not pat_colors: return grid
    
    h_divs = [r for r in range(R) if np.all(g[r]==wall)]
    v_divs = [c for c in range(C) if np.all(g[:,c]==wall)]
    
    def get_bands(divs, size):
        bands=[]; prev=-1
        for d in divs:
            if d>prev+1: bands.append((prev+1,d-1))
            prev=d
        if prev<size-1: bands.append((prev+1,size-1))
        return bands
    
    row_bands = get_bands(h_divs, R)
    col_bands = get_bands(v_divs, C)
    
    out = g.copy()
    
    for pat_color in pat_colors:
        orig_ri, orig_ci = None, None
        orig_pattern = None
        for ri,(r1,r2) in enumerate(row_bands):
            for ci,(c1,c2) in enumerate(col_bands):
                sub = g[r1:r2+1, c1:c2+1]
                if np.any(sub==pat_color):
                    orig_ri, orig_ci = ri, ci
                    orig_pattern = (sub==pat_color)
                    break
            if orig_ri is not None: break
        
        if orig_pattern is None: continue
        row_parity = orig_ri % 2
        col_parity = orig_ci % 2
        
        for ri,(r1,r2) in enumerate(row_bands):
            for ci,(c1,c2) in enumerate(col_bands):
                if ri%2==row_parity and ci%2==col_parity:
                    h = r2-r1+1; w = c2-c1+1
                    ph,pw = orig_pattern.shape
                    p = orig_pattern[:min(ph,h), :min(pw,w)]
                    out[r1:r1+p.shape[0], c1:c1+p.shape[1]][p] = pat_color
    
    return out.tolist()
