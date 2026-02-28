
def transform(grid):
    import numpy as np
    from scipy.ndimage import label
    g = np.array(grid)
    result = g.copy()
    rows, cols = g.shape
    bg = 0
    # Find connected components of 1s
    mask = (g != bg)
    labeled, n = label(mask)
    layer_colors = [1, 2, 3, 2, 1, 2, 3, 2, 1]
    for i in range(1, n+1):
        comp_mask = (labeled == i)
        rs = np.where(comp_mask.any(axis=1))[0]
        cs = np.where(comp_mask.any(axis=0))[0]
        if len(rs) == 0 or len(cs) == 0:
            continue
        r0,r1 = rs[0],rs[-1]
        c0,c1 = cs[0],cs[-1]
        h = r1-r0+1
        w = c1-c0+1
        # For each cell, determine layer = min distance to edge of bounding box
        for rr in range(r0, r1+1):
            for cc in range(c0, c1+1):
                if not comp_mask[rr, cc]:
                    continue
                dist = min(rr-r0, r1-rr, cc-c0, c1-cc)
                color = layer_colors[dist % len(layer_colors)]
                result[rr, cc] = color
    return result.tolist()
