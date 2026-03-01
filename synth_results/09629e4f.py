import numpy as np

def transform(grid):
    inp = np.array(grid)
    rows, cols = inp.shape
    sep_rows = set(r for r in range(rows) if all(inp[r,c]==5 for c in range(cols)))
    sep_cols = set(c for c in range(cols) if all(inp[r,c]==5 for r in range(rows)))
    
    non_sep_rows = sorted(set(range(rows)) - sep_rows)
    non_sep_cols = sorted(set(range(cols)) - sep_cols)
    
    def build_groups(lst):
        groups = []
        cur = []
        for x in lst:
            if cur and x != cur[-1]+1: groups.append(cur); cur = [x]
            else: cur.append(x)
        if cur: groups.append(cur)
        return groups
    
    row_groups = build_groups(non_sep_rows)
    col_groups = build_groups(non_sep_cols)
    
    # Find subgrid with fewest unique non-zero values (missing one value)
    special = None
    min_vals = 999
    subgrids = {}
    for sr, rg in enumerate(row_groups):
        for sc, cg in enumerate(col_groups):
            sub = inp[np.ix_(rg, cg)]
            nz_vals = set(sub[sub!=0].tolist())
            subgrids[(sr,sc)] = (rg, cg, sub, nz_vals)
            if len(nz_vals) < min_vals:
                min_vals = len(nz_vals)
                special = (sr, sc)
    
    rg_s, cg_s, sub_s, _ = subgrids[special]
    result = inp.copy().astype(int)
    
    for sr2, rg2 in enumerate(row_groups):
        for sc2, cg2 in enumerate(col_groups):
            meta_val = sub_s[sr2, sc2]
            for r in rg2:
                for c in cg2:
                    result[r, c] = meta_val
    
    return result.tolist()
