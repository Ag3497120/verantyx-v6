def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    sep_cols = [c for c in range(cols) if all(grid[r][c]==0 for r in range(rows))]
    block_starts = [0] + [c+1 for c in sep_cols]
    block_ends = sep_cols + [cols]
    
    pattern_map = {
        ((1,1),(1,2),(2,1),(2,2)): 8,
        (): 2,
        ((2,1),(2,2),(3,1),(3,2)): 4,
        ((1,0),(1,3),(2,0),(2,3)): 3,
    }
    
    result = []
    for bi, (bs, be) in enumerate(zip(block_starts, block_ends)):
        block = [[grid[r][c] for c in range(bs, be)] for r in range(rows)]
        zeros = tuple((r,c) for r in range(rows) for c in range(be-bs) if block[r][c]==0)
        val = pattern_map.get(zeros, 2)
        result.append([val]*3)
    
    return result
