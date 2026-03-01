from collections import Counter

def transform(grid):
    rows = len(grid); cols = len(grid[0])
    flat = [(r,c,grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c]!=0]
    block_val = None; block_r = None; block_c = None
    for r in range(rows-1):
        for c in range(cols-1):
            if (grid[r][c]!=0 and grid[r][c]==grid[r+1][c] and
                grid[r][c]==grid[r][c+1] and grid[r][c]==grid[r+1][c+1]):
                block_val = grid[r][c]; block_r, block_c = r, c; break
        if block_val: break
    singles = [(r,c,v) for r,c,v in flat if v != block_val]
    corners = [(block_r, block_c), (block_r, block_c+1), (block_r+1, block_c), (block_r+1, block_c+1)]
    result_vals = [0,0,0,0]
    for r,c,v in singles:
        dists = [abs(r-cr)+abs(c-cc) for cr,cc in corners]
        idx = dists.index(min(dists))
        result_vals[idx] = v
    result = [[0]*cols for _ in range(rows)]
    for i, (cr,cc) in enumerate(corners):
        result[cr][cc] = result_vals[i]
    return result
