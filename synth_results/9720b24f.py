def transform(grid):
    import numpy as np
    from scipy.ndimage import label
    from scipy.spatial import ConvexHull, Delaunay
    g = np.array(grid)
    R, C = g.shape
    
    out = g.copy()
    
    # Get positions for each color
    color_pos = {}
    for v in np.unique(g):
        if v == 0: continue
        color_pos[int(v)] = np.argwhere(g == v).astype(float)
    
    def in_hull(points, hull_pts):
        """Check if any of points are inside convex hull of hull_pts"""
        if len(hull_pts) < 3:
            return False
        try:
            hull = Delaunay(hull_pts)
            return any(hull.find_simplex(p) >= 0 for p in points)
        except:
            return False
    
    # Find all connected components
    all_components = []
    for v in np.unique(g):
        if v == 0: continue
        labeled, num = label(g == v)
        for cid in range(1, num+1):
            pos = np.argwhere(labeled == cid).astype(float)
            size = len(pos)
            mask = labeled == cid
            all_components.append((int(v), size, pos, mask))
    
    color_count = {int(v): int((g==v).sum()) for v in np.unique(g) if v != 0}
    
    for vi, si, pos_i, maski in all_components:
        for vj, sj in color_count.items():
            if vi == vj: continue
            if sj > si:
                hull_pts = color_pos[vj]
                if in_hull(pos_i, hull_pts):
                    out[maski] = 0
                    break
    
    return out.tolist()
