def transform(grid):
    rows,cols=len(grid),len(grid[0])
    # Find 5-separator column
    sep=None
    for c in range(cols):
        if all(grid[r][c]==5 for r in range(rows)): sep=c; break
    if sep is None: return grid
    left=[[grid[r][c] for c in range(sep)] for r in range(rows)]
    right=[[grid[r][c] for c in range(sep+1,cols)] for r in range(rows)]
    # Find the hollow rectangle in left (border color)
    from collections import Counter
    bc=Counter(v for row in left for v in row if v!=0).most_common(1)
    if not bc: return grid
    bc=bc[0][0]
    cells=[(r,c) for r in range(rows) for c in range(len(left[r])) if left[r][c]==bc]
    if not cells: return grid
    min_r=min(r for r,c in cells); max_r=max(r for r,c in cells)
    min_c=min(c for r,c in cells); max_c=max(c for r,c in cells)
    # find non-zero content in right
    right_nz=[(r,c,right[r][c]) for r in range(rows) for c in range(len(right[r])) if right[r][c]!=0]
    out=[list(row) for row in left]
    if right_nz:
        rrs=[r for r,c,v in right_nz]; rcs=[c for r,c,v in right_nz]
        rmin_r=min(rrs); rmin_c=min(rcs)
        for r,c,v in right_nz:
            nr=min_r+1+(r-rmin_r); nc=min_c+1+(c-rmin_c)
            if min_r<nr<max_r and min_c<nc<max_c: out[nr][nc]=v
    return out
