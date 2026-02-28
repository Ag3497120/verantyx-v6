def transform(grid):
    rows,cols=len(grid),len(grid[0])
    g=[list(row) for row in grid]
    bar_starts={}
    for c in range(cols):
        s=[r for r in range(rows) if grid[r][c]==2]
        if s: bar_starts[c]=min(s)
    if not bar_starts: return grid
    sv=sorted(set(bar_starts.values()))
    if len(sv)<2: return grid
    min_s=sv[0]; thr=sv[1]
    for c,s in bar_starts.items():
        if s==min_s:
            for r in range(thr,rows):
                if g[r][c]==2: g[r][c]=8
    return g
