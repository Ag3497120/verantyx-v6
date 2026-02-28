
def transform(grid):
    import numpy as np
    from scipy.ndimage import label
    from collections import Counter
    g = np.array(grid)
    rows, cols = g.shape
    # Detect background
    cnt = Counter(g.flatten().tolist())
    bg = cnt.most_common(1)[0][0]
    result = g.copy()
    non_bg = (g != bg)
    labeled, n = label(non_bg)
    for i in range(1, n+1):
        comp = (labeled == i)
        rs = np.where(comp.any(axis=1))[0]
        cs = np.where(comp.any(axis=0))[0]
        if len(rs) == 0 or len(cs) == 0:
            continue
        r0,r1 = rs[0],rs[-1]
        c0,c1 = cs[0],cs[-1]
        color = g[r0, c0]
        # Clear blob
        result[r0:r1+1, c0:c1+1] = bg
        # Shrink by 1
        nr0, nr1 = r0+1, r1-1
        nc0, nc1 = c0+1, c1-1
        if nr0 <= nr1 and nc0 <= nc1:
            result[nr0:nr1+1, nc0:nc1+1] = color
    return result.tolist()
