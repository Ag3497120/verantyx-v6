def transform(grid):
    return _solve(grid)

def solve_7d18a6fb(grid):
    g = np.array(grid)
    H, W = g.shape
    
    # Find the key grid (region with bg=1 containing some color labels)
    # The key grid has positions for each shape color
    # Find regions with 1s background
    key_region = None
    key_r0, key_c0 = None, None
    
    # Look for a rectangular region of 1s
    for r in range(H):
        for c in range(W):
            if g[r, c] == 1:
                # Check if this is part of a rectangular key
                # Find extent of 1s region from here
                r1, c1 = r, c
                while r1 < H and g[r1, c] == 1: r1 += 1
                c1 = c
                while c1 < W and g[r, c1] == 1: c1 += 1
                if r1-r > 2 and c1-c > 2:
                    # Check if it's a solid rectangle of 1s with embedded colors
                    region = g[r:r1, c:c1]
                    if all(region[0, :] == 1) and all(region[-1, :] == 1) and all(region[:, 0] == 1) and all(region[:, -1] == 1):
                        key_region = region
                        key_r0, key_c0 = r, c
                        break
        if key_region is not None: break
    
    if key_region is None: return grid
    
    H_key, W_key = key_region.shape
    
    # Find color anchor positions in key (non-1 cells)
    anchors = {}
    for r in range(H_key):
        for c in range(W_key):
            if key_region[r, c] != 1:
                anchors[int(key_region[r, c])] = (r, c)
    
    # Find shape clusters in the rest of the grid
    shapes = {}
    non_bg_non_1 = set(g.flatten()) - {0, 1}
    for color in non_bg_non_1:
        positions = list(zip(*np.where(g == color)))
        if not positions: continue
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        r0, r1 = min(rows), max(rows)+1
        c0, c1 = min(cols), max(cols)+1
        # Exclude key region positions
        shape_in_key = any(key_r0 <= p[0] < key_r0+H_key and key_c0 <= p[1] < key_c0+W_key for p in positions)
        if not shape_in_key:
            shapes[color] = g[r0:r1, c0:c1].tolist()
    
    # Create output grid
    out_H, out_W = H_key, W_key
    result = [[0]*out_W for _ in range(out_H)]
    
    for color, (anchor_r, anchor_c) in anchors.items():
        if color not in shapes: continue
        shape = shapes[color]
        sh, sw = len(shape), len(shape[0])
        # Place shape centered on anchor
        r_start = anchor_r - sh//2
        c_start = anchor_c - sw//2
        for dr in range(sh):
            for dc in range(sw):
                nr, nc = r_start+dr, c_start+dc
                if 0<=nr<out_H and 0<=nc<out_W and shape[dr][dc] != 0:
                    result[nr][nc] = shape[dr][dc]
    
    return result


_solve = solve_7d18a6fb
