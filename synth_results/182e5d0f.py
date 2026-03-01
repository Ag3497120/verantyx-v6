def transform(grid):
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for ar in range(rows):
        for ac in range(cols):
            if grid[ar][ac] != 3: continue
            has_zero = any(0<=ar+dr<rows and 0<=ac+dc<cols and grid[ar+dr][ac+dc]==0
                          for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)])
            if not has_zero: continue
            adj3 = [(ar+dr,ac+dc) for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                    if 0<=ar+dr<rows and 0<=ac+dc<cols and grid[ar+dr][ac+dc]==3]
            if len(adj3) != 1: continue
            first_r, first_c = adj3[0]
            path = []
            cr,cc = first_r, first_c
            pr,pc = ar,ac
            while 0<=cr<rows and 0<=cc<cols and grid[cr][cc] == 3:
                path.append((cr,cc))
                found_next = False
                for dr2,dc2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr2,nc2 = cr+dr2,cc+dc2
                    if (nr2,nc2) == (pr,pc): continue
                    if 0<=nr2<rows and 0<=nc2<cols and grid[nr2][nc2] == 3:
                        pr,pc = cr,cc; cr,cc = nr2,nc2; found_next = True; break
                if not found_next: break
            if not path: continue
            end_r, end_c = path[-1]
            five_r, five_c = None, None
            for dr2,dc2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr2,nc2 = end_r+dr2, end_c+dc2
                if 0<=nr2<rows and 0<=nc2<cols and grid[nr2][nc2] == 5:
                    five_r, five_c = nr2, nc2; break
            if five_r is None: continue
            result[path[0][0]][path[0][1]] = 5
            for pr2,pc2 in path[1:]:
                result[pr2][pc2] = 7
            result[five_r][five_c] = 7
    return result
