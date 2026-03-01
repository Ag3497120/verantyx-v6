def transform(grid):
    import numpy as np
    from collections import Counter, deque
    
    g = np.array(grid)
    rows, cols = g.shape
    
    border = g[0, 0]
    sec_size = rows - 2
    
    div_cols = [c for c in range(cols) if (g[:, c] == border).sum() >= rows - 1]
    
    def get_section(idx):
        c_start = div_cols[idx] + 1
        return g[1:1+sec_size, c_start:c_start+sec_size].copy()
    
    sec1 = get_section(0)
    sec2 = get_section(1)
    sec3 = get_section(2)
    
    all_vals = [v for v in list(sec2.flatten()) + list(sec3.flatten()) if v != border]
    bg = Counter(all_vals).most_common(1)[0][0]
    
    sec1_vals = [v for v in sec1.flatten() if v != border and v != bg]
    if not sec1_vals:
        return grid
    special_color = Counter(sec1_vals).most_common(1)[0][0]
    
    walls = set((r, c) for r in range(sec_size) for c in range(sec_size) if sec1[r, c] == special_color)
    
    def special_neighbors_8(r, c):
        return sum(1 for dr in [-1,0,1] for dc in [-1,0,1] 
                   if (dr,dc)!=(0,0) and (r+dr, c+dc) in walls)
    
    extra_cells = set((r, c) for r, c in walls if special_neighbors_8(r, c) == 0)
    
    # BFS from background cells adjacent to extra cells to find sec2 region
    seeds = set()
    for er, ec in extra_cells:
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = er+dr, ec+dc
            if 0 <= nr < sec_size and 0 <= nc < sec_size and (nr, nc) not in walls:
                seeds.add((nr, nc))
    
    sec2_region = set()
    queue = deque(seeds)
    visited = set(seeds)
    while queue:
        r, c = queue.popleft()
        sec2_region.add((r, c))
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < sec_size and 0 <= nc < sec_size and (nr, nc) not in visited and (nr, nc) not in walls:
                visited.add((nr, nc))
                queue.append((nr, nc))
    
    # Fill section 4
    sec4 = np.full((sec_size, sec_size), bg, dtype=int)
    for r in range(sec_size):
        for c in range(sec_size):
            if (r, c) in walls:
                if (r, c) in extra_cells:
                    # Extra cells: always show sec2 (including bg)
                    sec4[r, c] = sec2[r, c]
                else:
                    # Main path cells: sec2 if non-bg, else sec3
                    sec4[r, c] = sec2[r, c] if sec2[r, c] != bg else sec3[r, c]
            elif (r, c) in sec2_region:
                sec4[r, c] = sec2[r, c]
            else:
                sec4[r, c] = sec3[r, c]
    
    out = g.copy()
    c_start4 = div_cols[3] + 1
    out[1:1+sec_size, c_start4:c_start4+sec_size] = sec4
    
    return out.tolist()
