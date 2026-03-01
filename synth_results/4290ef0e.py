
def transform(grid):
    from collections import Counter, defaultdict
    rows, cols = len(grid), len(grid[0])
    bg = Counter(grid[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]
    by_color = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                by_color[grid[r][c]].append((r,c))
    def max_bbox_dim(cells):
        rs=[p[0] for p in cells]; cs=[p[1] for p in cells]
        return max(max(rs)-min(rs)+1, max(cs)-min(cs)+1)
    shapes = [(c, cells) for c,cells in by_color.items()]
    shapes.sort(key=lambda x: max_bbox_dim(x[1]), reverse=True)
    mx = max(max_bbox_dim(cells) for _,cells in shapes)
    if mx % 2 == 0: mx += 1
    out_size = mx
    out_center = out_size // 2
    result = [[bg]*out_size for _ in range(out_size)]
    for color, cells in shapes:
        rs=[p[0] for p in cells]; cs=[p[1] for p in cells]
        r1=min(rs); c1=min(cs); r2=max(rs); c2=max(cs)
        bbox_h = r2-r1+1; bbox_w = c2-c1+1
        d = max(bbox_h, bbox_w)
        shift_r = out_center - d//2
        shift_c = out_center - d//2
        # relative cells
        rel = [(r-r1, c-c1) for r,c in cells]
        # h-flip if right column heavier than left
        col_counts = [0]*bbox_w
        for rr,rc in rel: col_counts[rc] += 1
        if bbox_w > 1 and col_counts[-1] > col_counts[0]:
            rel = [(rr, bbox_w-1-rc) for rr,rc in rel]
        # v-flip if bottom row heavier than top
        row_counts = [0]*bbox_h
        for rr,rc in rel: row_counts[rr] += 1
        if bbox_h > 1 and row_counts[-1] > row_counts[0]:
            rel = [(bbox_h-1-rr, rc) for rr,rc in rel]
        # place with 4-fold reflection around output center
        to_place = set()
        for rr,rc in rel:
            or_ = shift_r + rr; oc = shift_c + rc
            to_place.add((or_,oc))
            to_place.add((out_size-1-or_,oc))
            to_place.add((or_,out_size-1-oc))
            to_place.add((out_size-1-or_,out_size-1-oc))
        for or_,oc in to_place:
            if 0<=or_<out_size and 0<=oc<out_size:
                result[or_][oc] = color
    return result
