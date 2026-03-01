def transform(grid):
    from collections import Counter, defaultdict
    R, C = len(grid), len(grid[0])
    bg = Counter(grid[r][c] for r in range(R) for c in range(C)).most_common(1)[0][0]
    nz = [(r,c,grid[r][c]) for r in range(R) for c in range(C) if grid[r][c]!=bg]
    if not nz: return [row[:] for row in grid]
    by_color = defaultdict(list)
    for r,c,v in nz:
        by_color[v].append((r,c))
    ref_len = len(by_color[9]) if 9 in by_color else None
    out = [row[:] for row in grid]
    def detect_groups(cells):
        anti_groups = defaultdict(list)
        main_groups = defaultdict(list)
        row_groups = defaultdict(list)
        col_groups = defaultdict(list)
        for r,c in cells:
            anti_groups[r+c].append((r,c))
            main_groups[r-c].append((r,c))
            row_groups[r].append((r,c))
            col_groups[c].append((r,c))
        options = [('anti',anti_groups),('main',main_groups),('row',row_groups),('col',col_groups)]
        return min(options, key=lambda x: len(x[1]))
    if ref_len is not None:
        for color, cells in by_color.items():
            if color == 9: continue
            gtype, groups = detect_groups(cells)
            for key, seg in groups.items():
                seg.sort()
                cur_len = len(seg)
                if cur_len == ref_len: continue
                if gtype == 'anti':
                    if cur_len < ref_len:
                        r0,c0 = seg[0]
                        for i in range(1, ref_len-cur_len+1):
                            nr,nc = r0-i,c0+i
                            if 0<=nr<R and 0<=nc<C: out[nr][nc]=color
                    else:
                        for r,c in seg[:cur_len-ref_len]: out[r][c]=bg
                elif gtype == 'main':
                    if cur_len < ref_len:
                        r0,c0 = seg[0]
                        for i in range(1, ref_len-cur_len+1):
                            nr,nc = r0-i,c0-i
                            if 0<=nr<R and 0<=nc<C: out[nr][nc]=color
                    else:
                        for r,c in seg[:cur_len-ref_len]: out[r][c]=bg
                elif gtype == 'row':
                    rr = seg[0][0]
                    cs = sorted(c for r,c in seg)
                    if cur_len < ref_len:
                        for c in range(cs[-1]+1, cs[-1]+ref_len-cur_len+1):
                            if 0<=c<C: out[rr][c]=color
                    else:
                        for c in cs[ref_len:]: out[rr][c]=bg
                elif gtype == 'col':
                    cc = seg[0][1]
                    rs = sorted(r for r,c in seg)
                    if cur_len < ref_len:
                        for r in range(rs[-1]+1, rs[-1]+ref_len-cur_len+1):
                            if 0<=r<R: out[r][cc]=color
                    else:
                        for r in rs[ref_len:]: out[r][cc]=bg
    else:
        col_cells = defaultdict(list)
        for r,c,v in nz:
            col_cells[c].append(r)
        if not col_cells: return out
        spans = sorted(len(rows) for rows in col_cells.values())
        med = spans[len(spans)//2]
        for c, rows in col_cells.items():
            rows.sort()
            color = grid[rows[0]][c]
            cur = len(rows)
            if cur < med:
                for r in range(rows[-1]+1, rows[-1]+med-cur+1):
                    if 0<=r<R: out[r][c]=color
            elif cur > med:
                for r in rows[med:]: out[r][c]=bg
    return out
