
def transform(grid):
    R,C=len(grid),len(grid[0])
    result=[row[:] for row in grid]
    r8=c8=None
    for r in range(R):
        for c in range(C):
            if grid[r][c]==8: r8,c8=r,c
    if r8 is None: return grid
    # Go UP
    r,c=r8-1,c8
    while 0<=r<R and 0<=c<C:
        result[r][c]=5
        r-=1
        if 0<=r<R:
            for dc in range(3):
                cc=c+dc
                if 0<=cc<C: result[r][cc]=5
            c+=2; r-=1
        else: break
    # Go DOWN
    r,c=r8+1,c8
    while 0<=r<R:
        if 0<=c<C: result[r][c]=5
        r+=1
        if 0<=r<R:
            for dc in range(3):
                cc=c-dc
                if 0<=cc<C: result[r][cc]=5
            c-=2; r+=1
        else: break
    return result
