
def transform(grid):
    R,C=len(grid),len(grid[0])
    result=[row[:] for row in grid]
    sep_rows=[r for r in range(R) if all(grid[r][c]==0 for c in range(C))]
    sep_cols=[c for c in range(C) if all(grid[r][c]==0 for r in range(R))]
    if not sep_rows or not sep_cols: return grid
    row_sec=[]
    prev=0
    for sr in sep_rows:
        if sr>prev: row_sec.append((prev,sr-1))
        prev=sr+1
    if prev<R: row_sec.append((prev,R-1))
    col_sec=[]
    prev=0
    for sc in sep_cols:
        if sc>prev: col_sec.append((prev,sc-1))
        prev=sc+1
    if prev<C: col_sec.append((prev,C-1))
    nr,nc=len(row_sec),len(col_sec)
    # Find dominant bg
    from collections import Counter
    flat=[grid[r][c] for r in range(R) for c in range(C)]
    bg=Counter(flat).most_common(1)[0][0]
    # Find special values per cell
    from collections import defaultdict
    val_pos=defaultdict(list)
    for ri,(r1,r2) in enumerate(row_sec):
        for ci,(c1,c2) in enumerate(col_sec):
            for r in range(r1,r2+1):
                for c in range(c1,c2+1):
                    v=grid[r][c]
                    if v!=bg and v!=0:
                        ir,ic=r-r1,c-c1
                        val_pos[v].append((ri,ci,ir,ic))
    # Propagate
    for v,positions in val_pos.items():
        row_ct=Counter(ri for ri,ci,ir,ic in positions)
        col_ct=Counter(ci for ri,ci,ir,ic in positions)
        for ri,cnt in row_ct.items():
            if cnt>=2:
                pts=[(ir,ic) for pli,pci,ir,ic in positions if pli==ri]
                ir0,ic0=pts[0]
                for ci2 in range(nc):
                    c1,c2=col_sec[ci2]; r1,r2=row_sec[ri]
                    if r1+ir0<=r2 and c1+ic0<=c2:
                        result[r1+ir0][c1+ic0]=v
        for ci,cnt in col_ct.items():
            if cnt>=2:
                pts=[(ir,ic) for pli,pci,ir,ic in positions if pci==ci]
                ir0,ic0=pts[0]
                for ri2 in range(nr):
                    c1,c2=col_sec[ci]; r1,r2=row_sec[ri2]
                    if r1+ir0<=r2 and c1+ic0<=c2:
                        result[r1+ir0][c1+ic0]=v
    return result
