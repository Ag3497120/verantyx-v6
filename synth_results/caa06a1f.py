def transform(grid):
    rows,cols=len(grid),len(grid[0])
    # Find non-border area (solid single-color regions are border)
    # Find repeating period in rows/cols
    from collections import Counter
    all_vals=Counter(v for row in grid for v in row)
    # Find which value is the border (appears in solid blocks)
    # Approach: scan row 0, find where it stops being periodic
    row0=list(grid[0])
    # Detect the sequence of colors before hitting a uniform block
    seq=[]
    solid_start=cols
    for c in range(cols):
        v=row0[c]
        if c>=1 and v==row0[c-1]:
            # potential solid block start
            if all(row0[cc]==v for cc in range(c,cols)):
                solid_start=c; break
        if len(seq)<10: seq.append(v)
    # Find period
    period=1
    for p in range(1,min(solid_start,len(seq))):
        if all(seq[i]==seq[i%p] for i in range(len(seq))):
            period=p; break
    # Colors in order (one period)
    colors=[seq[i%period] for i in range(period)]
    # Output: shift by 1
    def cell(r,c):
        idx=(r+c)%period
        return colors[(idx+1)%period]
    return [[cell(r,c) for c in range(cols)] for r in range(rows)]
