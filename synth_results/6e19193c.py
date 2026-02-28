def transform(grid):
    return _solve(grid)

def solve_6e19193c(grid):
    g = [list(r) for r in grid]
    H, W = len(g), len(g[0])
    # Find L-shaped objects (3 cells each)
    non_zero = [(r,c) for r in range(H) for c in range(W) if g[r][c] != 0]
    # Find connected components
    visited = set()
    components = []
    for (r,c) in non_zero:
        if (r,c) not in visited:
            comp = []
            q = [(r,c)]
            while q:
                cr, cc = q.pop()
                if (cr,cc) in visited: continue
                visited.add((cr,cc))
                comp.append((cr,cc))
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc = cr+dr,cc+dc
                    if 0<=nr<H and 0<=nc<W and g[nr][nc]!=0 and (nr,nc) not in visited:
                        q.append((nr,nc))
            components.append(comp)
    
    for comp in components:
        if len(comp) != 3: continue
        # Find the 2x2 bounding box
        rows = [c[0] for c in comp]
        cols = [c[1] for c in comp]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        if max_r-min_r != 1 or max_c-min_c != 1: continue
        # Find open corner = cell in 2x2 bbox not in comp
        bbox_cells = {(r,c) for r in range(min_r,max_r+1) for c in range(min_c,max_c+1)}
        comp_set = set(comp)
        open_corners = list(bbox_cells - comp_set)
        if len(open_corners) != 1: continue
        oc_r, oc_c = open_corners[0]
        # Find elbow = the cell that connects the two arms
        # The elbow is the cell NOT in open_corner's row and NOT in open_corner's col
        # Actually: elbow is the cell adjacent to both arms
        # From elbow, direction = from elbow through open corner and beyond
        for cell in comp:
            # Check if removing this cell disconnects the other two
            others = [c for c in comp if c != cell]
            r1,c1 = others[0]; r2,c2 = others[1]
            if abs(r1-r2)+abs(c1-c2) == 1:
                # This cell is the elbow
                er, ec = cell
                # Direction from elbow toward open corner
                dr, dc = oc_r-er, oc_c-ec
                # Draw diagonal from elbow in this direction, skip inside bbox
                nr, nc = er+dr, ec+dc
                while 0<=nr<H and 0<=nc<W:
                    # Skip if inside bbox
                    if not (min_r<=nr<=max_r and min_c<=nc<=max_c):
                        g[nr][nc] = g[er][ec]  # same color as elbow
                    nr += dr; nc += dc
                break
    return g


_solve = solve_6e19193c
