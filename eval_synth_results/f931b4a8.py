def transform(grid):
    H = len(grid); W = len(grid[0])
    qh = H // 2; qw = W // 2
    TL = [grid[r][:qw] for r in range(qh)]
    TR = [grid[r][qw:] for r in range(qh)]
    BL = [grid[r][:qw] for r in range(qh, H)]
    BR = [grid[r][qw:] for r in range(qh, H)]
    
    out_H = sum(1 for r in range(qh) for c in range(qw) if TL[r][c] != 0)
    out_W = sum(1 for r in range(qh) for c in range(qw) if TR[r][c] != 0)
    
    result = []
    for r in range(out_H):
        row = []
        for c in range(out_W):
            br_val = BR[r % qh][c % qw]
            if br_val != 0:
                row.append(br_val)
            else:
                row.append(BL[(r // qh) % qh][c % qw])
        result.append(row)
    return result
