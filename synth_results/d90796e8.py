def transform(grid):
    rows = len(grid); cols = len(grid[0])
    result = [row[:] for row in grid]
    merged_threes = set(); merged_twos = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3:
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc = r+dr,c+dc
                    if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]==2:
                        merged_threes.add((r,c)); merged_twos.add((nr,nc))
    for r,c in merged_threes: result[r][c] = 8
    for r,c in merged_twos: result[r][c] = 0
    return result
