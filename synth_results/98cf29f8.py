def transform(grid):
    import numpy as np
    from scipy.ndimage import label
    g = np.array(grid)
    R, C = g.shape
    
    # Find all non-zero connected components by color
    comps_by_color = {}
    for v in np.unique(g):
        if v == 0: continue
        labeled, num = label(g == v)
        for cid in range(1, num+1):
            pos = np.argwhere(labeled == cid)
            r1,c1 = pos.min(0); r2,c2 = pos.max(0)
            h,w = r2-r1+1, c2-c1+1
            density = len(pos) / (h*w)
            comps_by_color[int(v)] = comps_by_color.get(int(v), [])
            comps_by_color[int(v)].append({'pos':pos,'r1':int(r1),'r2':int(r2),'c1':int(c1),'c2':int(c2),'density':density,'mask':labeled==cid,'v':int(v)})
    
    # Identify target (solid rect) and mover (has arm structure)
    target = None
    mover = None
    for v, comps in comps_by_color.items():
        for comp in comps:
            if comp['density'] > 0.95:
                if target is None or (comp['r2']-comp['r1'])*(comp['c2']-comp['c1']) > (target['r2']-target['r1'])*(target['c2']-target['c1']):
                    target = comp
            else:
                if mover is None or len(comp['pos']) > len(mover['pos']):
                    mover = comp
    
    if not target or not mover:
        return grid
    
    out = g.copy()
    out[mover['mask']] = 0
    
    mv = mover['v']
    mask = mover['mask']
    mr1,mr2,mc1,mc2 = mover['r1'],mover['r2'],mover['c1'],mover['c2']
    tr1,tr2,tc1,tc2 = target['r1'],target['r2'],target['c1'],target['c2']
    
    # Determine arm direction
    tc_r = (tr1+tr2)/2; tc_c = (tc1+tc2)/2
    mc_r = (mr1+mr2)/2; mc_c = (mc1+mc2)/2
    dr = tc_r - mc_r; dc = tc_c - mc_c
    
    if abs(dc) >= abs(dr):
        # Horizontal arm
        arm_dir = 'right' if dc > 0 else 'left'
        # Body = cols present in ALL rows of mover
        mover_rows = [r for r in range(R) if np.any(mask[r,:])]
        body_cols = set(range(C))
        for r in mover_rows:
            body_cols &= set(np.where(mask[r,:])[0].tolist())
        body_cols = sorted(body_cols)
        body_rows = mover_rows
        
        body_h = len(body_rows)
        body_w = len(body_cols)
        body_r1 = min(body_rows); body_r2 = max(body_rows)
        
        if arm_dir == 'right':
            new_c1 = tc1 - body_w
            new_c2 = tc1 - 1
        else:
            new_c1 = tc2 + 1
            new_c2 = tc2 + body_w
        
        if 0 <= new_c1 and new_c2 < C:
            out[body_r1:body_r2+1, new_c1:new_c2+1] = mv
    else:
        # Vertical arm
        arm_dir = 'down' if dr > 0 else 'up'
        # Body = rows present in ALL cols of mover
        mover_cols = [c for c in range(C) if np.any(mask[:,c])]
        body_rows = set(range(R))
        for c in mover_cols:
            body_rows &= set(np.where(mask[:,c])[0].tolist())
        body_rows = sorted(body_rows)
        body_cols = mover_cols
        
        body_h = len(body_rows)
        body_w = len(body_cols)
        body_c1 = min(body_cols); body_c2 = max(body_cols)
        
        if arm_dir == 'up':
            new_r1 = tr2 + 1
            new_r2 = tr2 + body_h
        else:
            new_r1 = tr1 - body_h
            new_r2 = tr1 - 1
        
        if 0 <= new_r1 and new_r2 < R:
            out[new_r1:new_r2+1, body_c1:body_c2+1] = mv
    
    return out.tolist()
