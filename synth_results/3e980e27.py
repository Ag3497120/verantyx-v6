from collections import defaultdict, Counter

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    nz = {(r,c): grid[r][c] for r in range(rows) for c in range(cols) if grid[r][c] != 0}
    visited = set()
    clusters = []
    for (r,c) in nz:
        if (r,c) in visited: continue
        cluster = {}
        stack = [(r,c)]
        visited.add((r,c))
        while stack:
            cr,cc = stack.pop()
            cluster[(cr,cc)] = grid[cr][cc]
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    if dr==0 and dc==0: continue
                    nr,nc=cr+dr,cc+dc
                    if (nr,nc) in nz and (nr,nc) not in visited:
                        visited.add((nr,nc)); stack.append((nr,nc))
        clusters.append(cluster)
    templates = []
    standalone = defaultdict(list)
    for cluster in clusters:
        colors = set(cluster.values())
        if len(colors) == 1:
            for pos, col in cluster.items():
                standalone[col].append(pos)
        else:
            cnt = Counter(cluster.values())
            anchor_col = min(cnt, key=lambda c: cnt[c])
            shape_col = max(cnt, key=lambda c: cnt[c])
            anchor_pos = [(r,c) for (r,c),v in cluster.items() if v == anchor_col]
            if len(anchor_pos) == 1:
                ar, ac = anchor_pos[0]
                offsets = [(r-ar, c-ac) for (r,c),v in cluster.items() if v == shape_col]
                templates.append((ar, ac, anchor_col, offsets, shape_col))
    for (ar, ac, anchor_col, offsets, shape_col) in templates:
        reflect_dc = (anchor_col == 2)
        for (tr, tc) in standalone[anchor_col]:
            for (dr, dc) in offsets:
                sdc = -dc if reflect_dc else dc
                nr, nc = tr+dr, tc+sdc
                if 0<=nr<rows and 0<=nc<cols:
                    result[nr][nc] = shape_col
    return result
