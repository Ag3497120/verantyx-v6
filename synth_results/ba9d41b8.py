def transform(grid):
    from collections import Counter
    rows,cols=len(grid),len(grid[0])
    g=[list(row) for row in grid]
    bg=Counter(v for row in grid for v in row).most_common(1)[0][0]
    colors=set(v for row in grid for v in row if v!=bg)
    for color in colors:
        cells=[(r,c) for r in range(rows) for c in range(cols) if grid[r][c]==color]
        if not cells: continue
        min_r=min(r for r,c in cells); max_r=max(r for r,c in cells)
        min_c=min(c for r,c in cells); max_c=max(c for r,c in cells)
        if len(cells)==(max_r-min_r+1)*(max_c-min_c+1) and len(cells)>4:
            for r in range(min_r+1,max_r):
                for c in range(min_c+1,max_c):
                    lr=r-min_r-1; lc=c-min_c-1
                    if (lr+lc)%2==0: g[r][c]=bg
    return g
