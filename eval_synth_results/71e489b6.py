def transform(grid):
    import copy
    from collections import deque
    rows, cols = len(grid), len(grid[0])
    out = copy.deepcopy(grid)
    
    def get_comps(val):
        vis = [[False]*cols for _ in range(rows)]
        comps = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == val and not vis[r][c]:
                    comp = []
                    q = deque([(r,c)])
                    vis[r][c] = True
                    while q:
                        cr,cc = q.popleft()
                        comp.append((cr,cc))
                        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr,nc = cr+dr, cc+dc
                            if 0<=nr<rows and 0<=nc<cols and not vis[nr][nc] and grid[nr][nc]==val:
                                vis[nr][nc] = True
                                q.append((nr,nc))
                    comps.append(comp)
        return comps
    
    comps0 = get_comps(0)
    total = rows * cols
    THRESH = max(5, total * 0.05)
    normal0 = [set(c) for c in comps0 if len(c) >= THRESH]
    normal0_all = set().union(*normal0) if normal0 else set()
    
    anomaly0 = set()
    for comp in comps0:
        comp_set = set(comp)
        if comp_set <= normal0_all:
            for r,c in comp:
                zn = sum(1 for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                        if 0<=r+dr<rows and 0<=c+dc<cols and grid[r+dr][c+dc]==0)
                if zn == 1:
                    anomaly0.add((r,c))
        else:
            anomaly0.update(comp)
    
    for r,c in anomaly0:
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if dr==0 and dc==0: continue
                nr,nc = r+dr, c+dc
                if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in anomaly0:
                    out[nr][nc] = 7
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                zn = sum(1 for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                        if 0<=r+dr<rows and 0<=c+dc<cols and grid[r+dr][c+dc]==0)
                if zn >= 3:
                    out[r][c] = 0
    
    return out
