
def transform(grid):
    import numpy as np
    from collections import deque
    g = np.array(grid)
    result = g.copy()
    rows, cols = g.shape
    bg = 0
    fill = 8
    wall = 1
    # Find all 1-colored connected components
    from scipy.ndimage import label
    mask1 = (g == wall)
    labeled, n = label(mask1)
    for i in range(1, n+1):
        comp = (labeled == i)
        rs = np.where(comp.any(axis=1))[0]
        cs = np.where(comp.any(axis=0))[0]
        if len(rs) == 0:
            continue
        r0,r1 = rs[0],rs[-1]
        c0,c1 = cs[0],cs[-1]
        if r1-r0 < 2 or c1-c0 < 2:
            continue
        # Try to flood fill from center
        cr, cc = (r0+r1)//2, (c0+c1)//2
        if g[cr, cc] == bg:
            # BFS from center, blocked by 1s
            visited = set()
            q = deque([(cr, cc)])
            visited.add((cr, cc))
            to_fill = []
            while q:
                r, c = q.popleft()
                to_fill.append((r, c))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr2, nc2 = r+dr, c+dc
                    if (0 <= nr2 < rows and 0 <= nc2 < cols and 
                        (nr2, nc2) not in visited and g[nr2, nc2] != wall):
                        visited.add((nr2, nc2))
                        q.append((nr2, nc2))
            for r, c in to_fill:
                result[r, c] = fill
    return result.tolist()
