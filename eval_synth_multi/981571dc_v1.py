def transform(grid):
    n = len(grid)
    out = [row[:] for row in grid]
    
    # Fill 0s from transpose (output is transpose-symmetric)
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(n):
                if out[i][j] == 0 and out[j][i] != 0:
                    out[i][j] = out[j][i]
                    changed = True
    
    # For remaining 0s: find matching fully-determined rows
    for i in range(n):
        if any(out[i][j] == 0 for j in range(n)):
            known = {j: out[i][j] for j in range(n) if out[i][j] != 0}
            for k in range(n):
                if k == i or any(out[k][j] == 0 for j in range(n)):
                    continue
                if all(out[k][j] == v for j, v in known.items()):
                    for j in range(n):
                        if out[i][j] == 0:
                            out[i][j] = out[k][j]
                    break
    
    # Final transpose symmetry pass
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(n):
                if out[i][j] == 0 and out[j][i] != 0:
                    out[i][j] = out[j][i]
                    changed = True
    
    return out
