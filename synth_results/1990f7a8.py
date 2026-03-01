
def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find all non-zero cells
    nonzero = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] != 0]
    if not nonzero:
        return grid
    
    rs = [p[0] for p in nonzero]; cs = [p[1] for p in nonzero]
    mid_r = (min(rs) + max(rs)) / 2
    mid_c = (min(cs) + max(cs)) / 2
    
    # Split into quadrants
    quads = {(0,0):[], (0,1):[], (1,0):[], (1,1):[]}
    for r,c in nonzero:
        qr = 0 if r <= mid_r else 1
        qc = 0 if c <= mid_c else 1
        quads[(qr,qc)].append((r,c))
    
    # For each quadrant, extract bounding box content
    def extract_shape(cells):
        if not cells:
            return []
        r1 = min(p[0] for p in cells); r2 = max(p[0] for p in cells)
        c1 = min(p[1] for p in cells); c2 = max(p[1] for p in cells)
        cell_set = set(cells)
        return [[grid[r][c] if (r,c) in cell_set else 0 for c in range(c1,c2+1)] for r in range(r1,r2+1)]
    
    shapes = {}
    for qpos, cells in quads.items():
        shapes[qpos] = extract_shape(cells)
    
    # Determine shape size (all should be same)
    sizes = [(len(s), len(s[0]) if s else 0) for s in shapes.values() if s]
    if not sizes:
        return grid
    # Use max size
    sh = max(s[0] for s in sizes)
    sw = max(s[1] for s in sizes)
    
    def pad_shape(shape, h, w):
        result = [[0]*w for _ in range(h)]
        for r, row in enumerate(shape):
            for c, v in enumerate(row):
                if r < h and c < w:
                    result[r][c] = v
        return result
    
    q00 = pad_shape(shapes.get((0,0),[]), sh, sw)
    q01 = pad_shape(shapes.get((0,1),[]), sh, sw)
    q10 = pad_shape(shapes.get((1,0),[]), sh, sw)
    q11 = pad_shape(shapes.get((1,1),[]), sh, sw)
    
    # Assemble with separator
    result = []
    for r in range(sh):
        result.append(q00[r] + [0] + q01[r])
    result.append([0] * (sw*2+1))
    for r in range(sh):
        result.append(q10[r] + [0] + q11[r])
    
    return result
