
def transform(grid):
    import numpy as np
    from scipy import ndimage
    g = np.array(grid)
    h, w = g.shape
    from collections import Counter
    cnt = Counter(g.flatten().tolist())
    bg = cnt.most_common(1)[0][0]
    N = 2  # trim leftmost N columns of each shape's bounding box
    
    colors = set(g.flatten().tolist()) - {bg}
    out = g.copy()
    
    for color in colors:
        mask = (g == color).astype(int)
        lbl, n = ndimage.label(mask)
        for i in range(1, n+1):
            cells = list(zip(*np.where(lbl == i)))
            if not cells: continue
            min_col = min(c for r, c in cells)
            for r, c in cells:
                if c < min_col + N:
                    out[r, c] = bg
    return out.tolist()
