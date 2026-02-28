def transform(grid):
    rows,cols=len(grid),len(grid[0])
    R,C=2*rows,2*cols
    out=[[0]*C for _ in range(R)]
    copies=[]
    k=0
    while True:
        or_r=k*(rows-1); or_c=k*(cols-1)
        if or_r>=R or or_c>=C: break
        copies.append((or_r,or_c))
        k+=1
    for or_r,or_c in reversed(copies):
        for r in range(rows):
            for c in range(cols):
                nr=or_r+r; nc=or_c+c
                if nr<R and nc<C: out[nr][nc]=grid[r][c]
    return out
