
def transform(grid):
    R,C=len(grid),len(grid[0])
    result=[row[:] for row in grid]
    # Horizontal lines
    for r in range(R):
        c2s=[c for c in range(C) if grid[r][c]==2]
        if len(c2s)>=2:
            c1,c2=min(c2s),max(c2s)
            if all(grid[r][c]==2 for c in range(c1,c2+1)):
                L=c2-c1+1
                for d in range(1,L//2+1):
                    for sign in [-1,1]:
                        rr=r+sign*d
                        if 0<=rr<R:
                            for c in range(c1+d,c2-d+1):
                                if 0<=c<C and result[rr][c]==0:
                                    result[rr][c]=8
    # Vertical lines
    for c in range(C):
        r2s=[r for r in range(R) if grid[r][c]==2]
        if len(r2s)>=2:
            r1,r2=min(r2s),max(r2s)
            if all(grid[r][c]==2 for r in range(r1,r2+1)):
                L=r2-r1+1
                for d in range(1,L//2+1):
                    for sign in [-1,1]:
                        cc=c+sign*d
                        if 0<=cc<C:
                            for r in range(r1+d,r2-d+1):
                                if 0<=r<R and result[r][cc]==0:
                                    result[r][cc]=8
    return result
