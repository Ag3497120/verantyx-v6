
def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    from collections import Counter
    sep_rows = [r for r in range(rows) if all(grid[r][c]==0 for c in range(cols))]
    sep_cols = [c for c in range(cols) if all(grid[r][c]==0 for r in range(rows))]
    if len(sep_rows) < 2 or len(sep_cols) < 2: return grid
    row_segs = [(0,sep_rows[0]),(sep_rows[0]+1,sep_rows[1]),(sep_rows[1]+1,rows)]
    col_segs = [(0,sep_cols[0]),(sep_cols[0]+1,sep_cols[1]),(sep_cols[1]+1,cols)]
    blocks = []
    for ri,(r1,r2) in enumerate(row_segs):
        for ci,(c1,c2) in enumerate(col_segs):
            block = [grid[r][c1:c2] for r in range(r1,r2)]
            blocks.append(((ri,ci),block,r1,c1,r2,c2))
    all_vals = Counter(v for _,block,_,_,_,_ in blocks for row in block for v in row if v)
    vals = [v for v,_ in all_vals.most_common()]
    if len(vals) < 2: return grid
    major_val, minor_val = vals[0], vals[-1]
    def count_minor(block): return sum(v==minor_val for row in block for v in row)
    sorted_blocks = sorted(blocks, key=lambda x: count_minor(x[1]))
    out = [row[:] for row in grid]
    for ri in range(3):
        r1,r2 = row_segs[ri]
        for ci in range(3):
            c1,c2 = col_segs[ci]
            rank = ri*3+(2-ci) if major_val < minor_val else (2-ri)*3+ci
            src = sorted_blocks[rank][1]
            for r in range(r2-r1):
                for c in range(c2-c1):
                    out[r1+r][c1+c] = src[r][c]
    return out
