def transform(grid):
    from collections import Counter
    R, C = len(grid), len(grid[0])
    border = grid[0][0]
    sep_cols = [c for c in range(C) if all(grid[r][c]==border for r in range(R))]
    sep_rows = [r for r in range(R) if all(grid[r][c]==border for c in range(C))]
    if len(sep_cols) < 2 or len(sep_rows) < 2:
        return grid
    bw = sep_cols[1] - sep_cols[0]
    bh = sep_rows[1] - sep_rows[0]
    
    unique_copies = []
    for ri, sr in enumerate(sep_rows[:-1]):
        row_copies = []
        for ci, sc in enumerate(sep_cols[:-1]):
            copy = tuple(grid[sr+r][sc+c] for r in range(1,bh) for c in range(1,bw))
            row_copies.append((ci, copy))
        vote = Counter(c for _,c in row_copies)
        majority = vote.most_common(1)[0][0]
        unique = [(ci,c) for ci,c in row_copies if c != majority]
        if unique:
            unique_copies.append(unique[0][1])
        else:
            unique_copies.append(row_copies[0][1])
    
    result = []
    for ui, uc in enumerate(unique_copies):
        block = [[border]*(bw+1) for _ in range(bh+1)]
        k = 0
        for r in range(1, bh):
            for c in range(1, bw):
                block[r][c] = uc[k]; k += 1
        if ui == 0:
            result.extend(block)
        else:
            result.extend(block[1:])
    return result
