from collections import Counter

def transform(grid_list):
    grid = [row[:] for row in grid_list]
    H = len(grid)
    W = len(grid[0])
    border = grid[0][0]
    
    frame = None
    for v in grid[1]:
        if v != border:
            frame = v
            break
    if frame is None:
        return grid
    
    col_sections = []
    in_s = False
    for c in range(W):
        is_b = all(grid[r][c] == border for r in range(H))
        if not is_b and not in_s:
            start = c; in_s = True
        elif is_b and in_s:
            col_sections.append((start, c)); in_s = False
    if in_s: col_sections.append((start, W))
    
    row_sections = []
    in_s = False
    for r in range(H):
        is_b = all(grid[r][c] == border for c in range(W))
        if not is_b and not in_s:
            start = r; in_s = True
        elif is_b and in_s:
            row_sections.append((start, r)); in_s = False
    if in_s: row_sections.append((start, H))
    
    for rs, re in row_sections:
        for cs, ce in col_sections:
            cr_s, cr_e = rs + 1, re - 1
            cc_s, cc_e = cs + 1, ce - 1
            if cr_e <= cr_s or cc_e <= cc_s:
                continue
            ch = cr_e - cr_s
            cw = cc_e - cc_s
            
            found = False
            # Try candidates in order of increasing tile size
            candidates = []
            for rp in range(1, ch + 1):
                for cp in range(1, cw + 1):
                    candidates.append((rp * cp, rp, cp))
            candidates.sort()
            
            for _, rp, cp in candidates:
                # Skip trivial (full size) tile
                if rp == ch and cp == cw:
                    continue
                # Need enough reps for voting
                if ch // rp < 2 and cw // cp < 2:
                    continue
                
                tile = [[Counter() for _ in range(cp)] for _ in range(rp)]
                for r in range(ch):
                    for c in range(cw):
                        tile[r % rp][c % cp][grid[cr_s + r][cc_s + c]] += 1
                
                all_clear = True
                errors = 0
                for rr in range(rp):
                    for cc in range(cp):
                        t = tile[rr][cc]
                        total = sum(t.values())
                        if total == 0:
                            all_clear = False
                            break
                        mc_cnt = t.most_common(1)[0][1]
                        if mc_cnt / total < 0.75:
                            all_clear = False
                            break
                    if not all_clear:
                        break
                
                if all_clear:
                    for r in range(ch):
                        for c in range(cw):
                            mc = tile[r % rp][c % cp].most_common(1)[0][0]
                            if grid[cr_s + r][cc_s + c] != mc:
                                errors += 1
                    if errors > 0 and errors / (ch * cw) < 0.15:
                        # Apply this tile
                        for r in range(ch):
                            for c in range(cw):
                                mc = tile[r % rp][c % cp].most_common(1)[0][0]
                                grid[cr_s + r][cc_s + c] = mc
                        found = True
                        break
            
    return grid
