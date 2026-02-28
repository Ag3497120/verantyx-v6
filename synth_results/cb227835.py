def transform(grid):
    g=[list(row) for row in grid]
    rows,cols=len(g),len(g[0])
    eights=[(r,c) for r in range(rows) for c in range(cols) if g[r][c]==8]
    if len(eights)<2: return g
    (r1,c1),(r2,c2)=eights[0],eights[1]
    dr=r2-r1; dc=c2-c1
    steps=max(abs(dr),abs(dc))
    if steps==0: return g
    sr=(1 if dr>0 else -1) if dr!=0 else 0
    sc=(1 if dc>0 else -1) if dc!=0 else 0
    # Two parallel diagonal lines
    for i in range(1,steps):
        r=r1+i*sr; c=c1+i*sc
        if 0<=r<rows and 0<=c<cols: g[r][c]=3
        # Perpendicular offset  
        r2b=r1+i*sr+sc; c2b=c1+i*sc-sr
        if 0<=r2b<rows and 0<=c2b<cols: g[r2b][c2b]=3
    return g
