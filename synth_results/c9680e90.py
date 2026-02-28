def transform(grid):
    rows,cols=len(grid),len(grid[0])
    from collections import Counter
    nine_row=next((r for r in range(rows) if all(grid[r][c]==9 for c in range(cols))),None)
    if nine_row is None: return grid
    bg=Counter(v for row in grid for v in row if v!=9).most_common(1)[0][0]
    out=[list(row) for row in grid]
    # Upper half: move non-bg values DOWN by 1, but track which have been moved
    moved=set()
    for r in range(nine_row-1,-1,-1):
        for c in range(cols):
            if grid[r][c]!=bg and (r,c) not in moved:
                nr=r+1
                if nr<nine_row:
                    out[nr][c]=grid[r][c]
                    if (r,c) not in moved: out[r][c]=bg
                    moved.add((nr,c))
    # Lower half: move non-bg UP by 1
    moved2=set()
    for r in range(nine_row+1,rows):
        for c in range(cols):
            if grid[r][c]!=bg and (r,c) not in moved2:
                nr=r-1
                if nr>nine_row:
                    out[nr][c]=grid[r][c]
                    if (r,c) not in moved2: out[r][c]=bg
                    moved2.add((nr,c))
    return out
