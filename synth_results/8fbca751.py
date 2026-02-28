def transform(grid):
    import numpy as np
    from scipy.ndimage import label
    from collections import defaultdict
    g = np.array(grid)
    out = g.copy()
    labeled, n = label(g > 0)
    bboxes = []
    for i in range(1, n+1):
        mask = labeled == i
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        bboxes.append([int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())])
    
    def should_merge(b1, b2):
        r1,r2,c1,c2 = b1; r3,r4,c3,c4 = b2
        row_overlap = min(r2,r4) >= max(r1,r3)
        col_overlap = min(c2,c4) >= max(c1,c3)
        col_gap = max(c3-c2-1, c1-c4-1, 0)
        row_gap = max(r3-r2-1, r1-r4-1, 0)
        # Merge if adjacent in one dimension and overlapping in the other
        if col_gap <= 1 and row_overlap:
            return True
        if row_gap <= 1 and col_overlap:
            return True
        return False
    
    parent = list(range(len(bboxes)))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y): parent[find(x)] = find(y)
    
    changed = True
    while changed:
        changed = False
        groups = defaultdict(list)
        for i, b in enumerate(bboxes): groups[find(i)].append(b)
        gbbox = {k: [min(b[0] for b in grp), max(b[1] for b in grp), min(b[2] for b in grp), max(b[3] for b in grp)] for k, grp in groups.items()}
        gkeys = list(gbbox.keys())
        for i in range(len(gkeys)):
            for j in range(i+1, len(gkeys)):
                ki, kj = gkeys[i], gkeys[j]
                if find(ki) != find(kj) and should_merge(gbbox[ki], gbbox[kj]):
                    union(ki, kj); changed = True
    
    grp2 = defaultdict(list)
    for i in range(len(bboxes)): grp2[find(i)].append(bboxes[i])
    for grp in grp2.values():
        r1=min(b[0] for b in grp); r2=max(b[1] for b in grp)
        c1=min(b[2] for b in grp); c2=max(b[3] for b in grp)
        out[r1:r2+1,c1:c2+1] = np.where(g[r1:r2+1,c1:c2+1]==0, 2, g[r1:r2+1,c1:c2+1])
    return out.tolist()
