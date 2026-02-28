def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    result = g.copy()
    
    # Find 3 non-zero cells (triangle vertices)
    nz = [(r, c, int(g[r, c])) for r in range(rows) for c in range(cols) if g[r, c] != 0]
    if len(nz) != 3:
        return grid
    
    A = (nz[0][0], nz[0][1])
    B = (nz[1][0], nz[1][1])
    C = (nz[2][0], nz[2][1])
    vA, vB, vC = nz[0][2], nz[1][2], nz[2][2]
    
    # Find longest edge
    def dist2(p, q):
        return (p[0]-q[0])**2 + (p[1]-q[1])**2
    
    edges = [(A, B, vA, vB), (B, C, vB, vC), (A, C, vA, vC)]
    edges.sort(key=lambda e: -dist2(e[0], e[1]))
    
    P, Q, vP, vQ = edges[0]
    
    # Place cells at 1/3, 1/2, 2/3 along longest edge
    def lerp(p, q, t):
        return (round(p[0] + t*(q[0]-p[0])), round(p[1] + t*(q[1]-p[1])))
    
    p13 = lerp(P, Q, 1/3)
    p12 = lerp(P, Q, 1/2)
    p23 = lerp(P, Q, 2/3)
    
    if 0 <= p13[0] < rows and 0 <= p13[1] < cols:
        result[p13[0], p13[1]] = vP  # same color as P endpoint
    if 0 <= p12[0] < rows and 0 <= p12[1] < cols:
        result[p12[0], p12[1]] = 5   # midpoint color
    if 0 <= p23[0] < rows and 0 <= p23[1] < cols:
        result[p23[0], p23[1]] = vQ  # same color as Q endpoint
    
    # Centroid: colored by nearest vertex
    centroid = (round((A[0]+B[0]+C[0])/3), round((A[1]+B[1]+C[1])/3))
    verts = [(A, vA), (B, vB), (C, vC)]
    nearest = min(verts, key=lambda v: dist2(centroid, v[0]))
    if 0 <= centroid[0] < rows and 0 <= centroid[1] < cols:
        result[centroid[0], centroid[1]] = nearest[1]
    
    return result.tolist()
