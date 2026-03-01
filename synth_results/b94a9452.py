
def transform(grid):
    R, C = len(grid), len(grid[0])
    cells = [(r,c,grid[r][c]) for r in range(R) for c in range(C) if grid[r][c]!=0]
    if not cells: return grid
    rows=[r for r,c,v in cells]; cols=[c for r,c,v in cells]
    r1,r2,c1,c2=min(rows),max(rows),min(cols),max(cols)
    border_color=grid[r1][c1]
    inner_color=None
    for r,c,v in cells:
        if r1<r<r2 and c1<c<c2 and v!=border_color:
            inner_color=v; break
    if inner_color is None:
        return [[grid[r][c] for c in range(c1,c2+1)] for r in range(r1,r2+1)]
    out=[]
    for r in range(r1,r2+1):
        row=[]
        for c in range(c1,c2+1):
            v=grid[r][c]
            if v==border_color: row.append(inner_color)
            elif v==inner_color: row.append(border_color)
            else: row.append(v)
        out.append(row)
    return out
