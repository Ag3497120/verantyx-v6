def transform(grid):
    H = len(grid)
    W = len(grid[0])
    result = [row[:] for row in grid]
    
    frames = []
    for r in range(H):
        for c in range(W):
            fc = grid[r][c]
            if r > 0 and grid[r-1][c] == fc:
                continue
            if c > 0 and grid[r][c-1] == fc:
                continue
            c2 = c
            while c2+1 < W and grid[r][c2+1] == fc:
                c2 += 1
            if c2 - c < 2:
                continue
            r2 = r
            while r2+1 < H and grid[r2+1][c] == fc:
                r2 += 1
            if r2 - r < 2:
                continue
            ok = True
            for cc in range(c, c2+1):
                if grid[r2][cc] != fc:
                    ok = False; break
            if not ok: continue
            for rr in range(r, r2+1):
                if grid[rr][c2] != fc:
                    ok = False; break
            if not ok: continue
            frames.append((r, c, r2, c2, fc))
    
    frames.sort(key=lambda f: (f[2]-f[0])*(f[3]-f[1]))
    fixed = set()
    
    for r1, c1, r2, c2, fc in frames:
        interior_cells = set()
        for r in range(r1+1, r2):
            for c in range(c1+1, c2):
                interior_cells.add((r,c))
        if interior_cells & fixed:
            continue
        
        interior = []
        for r in range(r1+1, r2):
            row = []
            for c in range(c1+1, c2):
                row.append(grid[r][c])
            interior.append(row)
        
        ih = len(interior)
        iw = len(interior[0]) if ih > 0 else 0
        if ih == 0 or iw == 0:
            continue
        total = ih * iw
        
        best = None
        for ph in range(1, ih+1):
            for pw in range(1, iw+1):
                area = ph * pw
                if total // area < 3:
                    continue
                
                tile = [[0]*pw for _ in range(ph)]
                errors = 0
                for tr in range(ph):
                    for tc in range(pw):
                        counts = {}
                        n = 0
                        for rr in range(tr, ih, ph):
                            for cc in range(tc, iw, pw):
                                v = interior[rr][cc]
                                counts[v] = counts.get(v, 0) + 1
                                n += 1
                        best_v = max(counts, key=counts.get)
                        tile[tr][tc] = best_v
                        errors += n - counts[best_v]
                
                score = (errors, area)
                if best is None or score < best[:2]:
                    best = (errors, area, ph, pw, [row[:] for row in tile])
        
        if best and best[0] > 0:
            _, _, ph, pw, tile = best
            for r in range(ih):
                for c in range(iw):
                    result[r1+1+r][c1+1+c] = tile[r%ph][c%pw]
            fixed |= interior_cells
    
    return result
