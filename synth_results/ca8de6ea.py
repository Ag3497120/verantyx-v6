
def transform(grid):
    n=len(grid)
    out=[[0]*3 for _ in range(3)]
    h=n//2
    out[0][0]=grid[0][0]; out[0][2]=grid[0][n-1]
    out[2][0]=grid[n-1][0]; out[2][2]=grid[n-1][n-1]
    out[1][1]=grid[h][h]
    out[0][1]=grid[1][1]; out[1][2]=grid[1][n-2]
    out[1][0]=grid[n-2][1]; out[2][1]=grid[n-2][n-2]
    return out
