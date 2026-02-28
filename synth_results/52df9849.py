
def transform(grid):
    import numpy as np
    from collections import Counter
    g = np.array(grid)
    bg = Counter(g.flatten().tolist()).most_common(1)[0][0]
    colors = [c for c in set(g.flatten().tolist()) if c != bg]
    if len(colors) < 2:
        return g.tolist()
    # Find bounding boxes for each color
    bboxes = {}
    for c in colors:
        rows, cols = np.where(g == c)
        bboxes[c] = (rows.min(), rows.max(), cols.min(), cols.max())
    out = g.copy()
    # For each cell that shows color A, if it's in the bbox of color B (B!=A,bg),
    # replace it with B
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if g[r, c] != bg:
                curr = g[r, c]
                for other_c, (r0, r1, c0, c1) in bboxes.items():
                    if other_c != curr and r0 <= r <= r1 and c0 <= c <= c1:
                        out[r, c] = other_c
                        break
    return out.tolist()
