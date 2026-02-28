import numpy as np
from collections import Counter, deque
from scipy import ndimage

def solve_6e02f1e3(grid):
    flat = [v for row in grid for v in row]
    distinct = len(set(flat))
    if distinct == 1:
        return [[5,5,5],[0,0,0],[0,0,0]]
    elif distinct == 2:
        return [[5,0,0],[0,5,0],[0,0,5]]
    else:
        return [[0,0,5],[0,5,0],[5,0,0]]

def solve_6e19193c(grid):
    g = [list(r) for r in grid]
    H, W = len(g), len(g[0])
    # Find L-shaped objects (3 cells each)
    non_zero = [(r,c) for r in range(H) for c in range(W) if g[r][c] != 0]
    # Find connected components
    visited = set()
    components = []
    for (r,c) in non_zero:
        if (r,c) not in visited:
            comp = []
            q = [(r,c)]
            while q:
                cr, cc = q.pop()
                if (cr,cc) in visited: continue
                visited.add((cr,cc))
                comp.append((cr,cc))
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc = cr+dr,cc+dc
                    if 0<=nr<H and 0<=nc<W and g[nr][nc]!=0 and (nr,nc) not in visited:
                        q.append((nr,nc))
            components.append(comp)
    
    for comp in components:
        if len(comp) != 3: continue
        # Find the 2x2 bounding box
        rows = [c[0] for c in comp]
        cols = [c[1] for c in comp]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        if max_r-min_r != 1 or max_c-min_c != 1: continue
        # Find open corner = cell in 2x2 bbox not in comp
        bbox_cells = {(r,c) for r in range(min_r,max_r+1) for c in range(min_c,max_c+1)}
        comp_set = set(comp)
        open_corners = list(bbox_cells - comp_set)
        if len(open_corners) != 1: continue
        oc_r, oc_c = open_corners[0]
        # Find elbow = the cell that connects the two arms
        # The elbow is the cell NOT in open_corner's row and NOT in open_corner's col
        # Actually: elbow is the cell adjacent to both arms
        # From elbow, direction = from elbow through open corner and beyond
        for cell in comp:
            # Check if removing this cell disconnects the other two
            others = [c for c in comp if c != cell]
            r1,c1 = others[0]; r2,c2 = others[1]
            if abs(r1-r2)+abs(c1-c2) == 1:
                # This cell is the elbow
                er, ec = cell
                # Direction from elbow toward open corner
                dr, dc = oc_r-er, oc_c-ec
                # Draw diagonal from elbow in this direction, skip inside bbox
                nr, nc = er+dr, ec+dc
                while 0<=nr<H and 0<=nc<W:
                    # Skip if inside bbox
                    if not (min_r<=nr<=max_r and min_c<=nc<=max_c):
                        g[nr][nc] = g[er][ec]  # same color as elbow
                    nr += dr; nc += dc
                break
    return g

def solve_72322fa7(grid):
    g = np.array(grid)
    H, W = g.shape
    # Find template 8 (8 that has non-zero neighbors)
    # Collect all 8 positions and their colored neighborhoods
    eights = list(zip(*np.where(g==8)))
    # For each 8, find non-8/non-0 cells within ±3
    radius = 4
    pattern = {}  # 8_pos -> {(dr,dc): color}
    for r8, c8 in eights:
        r8, c8 = int(r8), int(c8)
        p = {}
        for dr in range(-radius, radius+1):
            for dc in range(-radius, radius+1):
                nr, nc = r8+dr, r8+dc if False else r8+dr, c8+dc
                nr2, nc2 = r8+dr, c8+dc
                if 0<=nr2<H and 0<=nc2<W and g[nr2,nc2] not in [0,8] and (dr,dc)!=(0,0):
                    p[(dr,dc)] = int(g[nr2,nc2])
        pattern[(r8,c8)] = p
    
    # Template 8s have patterns; lone 8s don't
    templates = {pos: p for pos, p in pattern.items() if p}
    lone_eights = {pos for pos, p in pattern.items() if not p}
    # Also find colored clusters without 8 (incomplete from other side)
    # For each template, get the pattern
    # Apply pattern to lone 8s
    out = g.copy()
    
    # Combine all templates (they should all have same shape)
    if templates:
        # Use first template's pattern
        all_patterns = list(templates.values())
        combined = {}
        for p in all_patterns:
            for k, v in p.items():
                combined[k] = v
        
        for (r8, c8) in lone_eights:
            for (dr, dc), color in combined.items():
                nr, nc = r8+dr, r8+dc if False else r8+dr, c8+dc
                nr2, nc2 = r8+dr, c8+dc
                if 0<=nr2<H and 0<=nc2<W:
                    out[nr2, nc2] = color
        
        # Find colored clusters without 8 (colored cells that match pattern offset from empty 8-spot)
        # For each template's pattern, check if the pattern exists somewhere without the central 8
        for (dr, dc), color in combined.items():
            # Find all occurrences of 'color' in grid
            color_positions = list(zip(*np.where(g==color)))
            for r_col, c_col in color_positions:
                r_col, c_col = int(r_col), int(c_col)
                # This color should be at template_8_pos + (dr,dc)
                # So potential 8 position = (r_col - dr, c_col - dc)
                r8_pot, c8_pot = r_col - dr, c_col - dc
                if 0<=r8_pot<H and 0<=c8_pot<W and g[r8_pot, c8_pot] == 0:
                    # Check if all other offsets in pattern also match
                    # (or just the subset around this potential 8)
                    match_count = 0
                    for (dr2, dc2), col2 in combined.items():
                        nr2, nc2 = r8_pot+dr2, c8_pot+dc2
                        if 0<=nr2<H and 0<=nc2<W and g[nr2,nc2] == col2:
                            match_count += 1
                    if match_count >= len(combined) - 1:  # most match
                        out[r8_pot, c8_pot] = 8
    
    return out.tolist()

def solve_72ca375d(grid):
    g = np.array(grid)
    colors = set(g.flatten()) - {0}
    best = None
    best_color = None
    for color in colors:
        positions = list(zip(*np.where(g==color)))
        if not positions: continue
        rows = [r for r,c in positions]
        cols = [c for r,c in positions]
        r0, r1, c0, c1 = min(rows), max(rows)+1, min(cols), max(cols)+1
        # Extract bbox region
        region = g[r0:r1, c0:c1]
        # Check LR symmetry (each row is palindrome)
        lr_sym = all((row == row[::-1]).all() for row in region)
        if lr_sym:
            best = (r0, r1, c0, c1)
            best_color = color
            break  # take first LR-symmetric one
    
    if best:
        r0, r1, c0, c1 = best
        # Return bbox region with only the matching color
        region = g[r0:r1, c0:c1].tolist()
        return region
    # Fallback: return most filled region
    return grid

def solve_73c3b0d8(grid):
    g = [list(r) for r in grid]
    H, W = len(g), len(g[0])
    # Find divider row (all 2s)
    div_row = -1
    for r in range(H):
        if all(g[r][c] == 2 for c in range(W)):
            div_row = r; break
    if div_row == -1: return grid
    
    out = [row[:] for row in g]
    # Clear all 4s above divider (they will be replaced)
    for r in range(div_row):
        for c in range(W):
            if out[r][c] == 4:
                out[r][c] = 0
    # Also clear 4s below divider
    for r in range(div_row+1, H):
        for c in range(W):
            if out[r][c] == 4:
                out[r][c] = 0
    
    # Process each 4
    original_fours_above = [(r,c) for r in range(div_row) for c in range(W) if g[r][c]==4]
    original_fours_below = [(r,c) for r in range(div_row+1,H) for c in range(W) if g[r][c]==4]
    
    # All 4s move 1 step down
    new_positions = {}
    for r, c in original_fours_above + original_fours_below:
        new_r = r + 1
        if 0 <= new_r < H:
            if (new_r, c) not in new_positions:
                new_positions[(new_r, c)] = 4
    
    # Place 4s and V-shapes
    for (new_r, new_c), val in new_positions.items():
        out[new_r][new_c] = 4
        # If this 4 is at divider-1 (row just above divider)
        if new_r == div_row - 1:
            # Generate V-shape going up from here
            step = 1
            while True:
                left_c = new_c - step
                right_c = new_c + step
                placed = False
                for vc, vr in [(left_c, new_r-step), (right_c, new_r-step)]:
                    if 0 <= vr < div_row and 0 <= vc < W:
                        out[vr][vc] = 4
                        placed = True
                if not placed: break
                step += 1
                if step > max(W, div_row): break
    
    return out

def solve_758abdf0(grid):
    g = [list(r) for r in grid]
    H, W = len(g), len(g[0])
    out = [row[:] for row in g]
    
    # Find 0-border: is it column 0 or row 0?
    col0_all_zero = all(g[r][0] == 0 for r in range(H))
    row0_all_zero = all(g[0][c] == 0 for c in range(W))
    
    if col0_all_zero:
        # Vertical orientation: 8s in rows, 0-border at col 0
        # For each row: find 8s in that row (excluding col 0)
        for r in range(H):
            row = g[r][1:]  # exclude col 0
            eights_cols = [c+1 for c, v in enumerate(row) if v == 8]
            if not eights_cols: continue
            # Count consecutive groups from left
            # Find the leftmost group of adjacent 8s
            run_start = eights_cols[0]
            run_end = run_start
            for ec in eights_cols[1:]:
                if ec == run_end + 1:
                    run_end = ec
                else:
                    break
            run_len = run_end - run_start + 1
            
            if run_len >= 2:
                # Double 8s -> remove and place 0,0 at end
                for c in range(run_start, run_end+1):
                    out[r][c] = 7 if g[r][c] == 8 else g[r][c]
                for c in range(run_start, run_end+1):
                    out[r][c] = 7
                # Actually find background value
                bg = g[r][2] if g[r][2] not in [0,8] else 7
                # Set run positions to bg
                for c in range(run_start, run_end+1):
                    out[r][c] = bg
                # Place 0,0 at end (last run_len cols)
                for i in range(run_len):
                    tc = W - 1 - i
                    if tc > 0:
                        out[r][tc] = 0
            else:
                # Single 8 -> extend to double
                if eights_cols[0]+1 < W and g[r][eights_cols[0]+1] != 8:
                    out[r][eights_cols[0]+1] = 8
    else:
        # Horizontal orientation: 8s in cols, 0-border at row 0
        for c in range(W):
            col = [g[r][c] for r in range(1, H)]
            eights_rows = [r+1 for r, v in enumerate(col) if v == 8]
            if not eights_rows: continue
            run_start = eights_rows[0]
            run_end = run_start
            for er in eights_rows[1:]:
                if er == run_end + 1:
                    run_end = er
                else:
                    break
            run_len = run_end - run_start + 1
            if run_len >= 2:
                bg = 7  # find bg
                for r in range(run_start, run_end+1):
                    out[r][c] = bg
                for i in range(run_len):
                    tr = H - 1 - i
                    if tr > 0:
                        out[tr][c] = 0
            else:
                if eights_rows[0]+1 < H and g[eights_rows[0]+1][c] != 8:
                    out[eights_rows[0]+1][c] = 8
    return out

def solve_762cd429(grid):
    g = [list(r) for r in grid]
    H, W = len(g), len(g[0])
    out = [row[:] for row in g]
    
    # Find the 2x2 seed (usually at rows r, r+1, cols 0, 1)
    seed = None
    seed_r, seed_c = None, None
    for r in range(H-1):
        for c in range(W-1):
            if any(grid[r+dr][c+dc] != 0 for dr in range(2) for dc in range(2)):
                if g[r][c] != 0 or g[r][c+1] != 0 or g[r+1][c] != 0 or g[r+1][c+1] != 0:
                    # Check it's the only non-zero 2x2 near left edge
                    if c < 3:
                        seed = [[g[r][c], g[r][c+1]], [g[r+1][c], g[r+1][c+1]]]
                        seed_r, seed_c = r, c
                        break
        if seed: break
    
    if not seed: return grid
    
    # Generate expansions
    center_r = seed_r  # top of seed
    start_col = seed_c + 2  # next scale starts after seed
    scale = 2  # each cell -> scale x scale block
    
    while start_col < W:
        # This level's block: size = scale x scale, placed at rows and cols
        block_size = scale
        start_r = center_r - block_size // 2 + 1  # center around seed
        
        for i in range(2):
            for j in range(2):
                cell_val = seed[i][j]
                # Fill block_size//2 x block_size//2 area
                half = block_size // 2
                r0 = start_r + i * half
                c0 = start_col + j * half
                for dr in range(half):
                    for dc in range(half):
                        nr, nc = r0+dr, c0+dc
                        if 0<=nr<H and 0<=nc<W:
                            out[nr][nc] = cell_val
        
        start_col += block_size
        scale *= 2
    
    return out

def solve_770cc55f(grid):
    g = [list(r) for r in grid]
    H, W = len(g), len(g[0])
    out = [row[:] for row in g]
    
    # Find divider row (all 2s)
    div_row = next(r for r in range(H) if all(g[r][c]==2 for c in range(W)))
    
    # Find pattern rows above and below
    # Pattern = non-zero non-2 columns
    top_pattern_row = None
    bot_pattern_row = None
    for r in range(div_row):
        if any(g[r][c] not in [0,2] for c in range(W)):
            top_pattern_row = r; break
    for r in range(H-1, div_row, -1):
        if any(g[r][c] not in [0,2] for c in range(W)):
            bot_pattern_row = r; break
    
    if top_pattern_row is None or bot_pattern_row is None: return grid
    
    top_cols = {c for c in range(W) if g[top_pattern_row][c] not in [0,2]}
    bot_cols = {c for c in range(W) if g[bot_pattern_row][c] not in [0,2]}
    
    # Intersection
    intersection = top_cols & bot_cols
    
    # Larger pattern determines which side gets filled
    if len(top_cols) >= len(bot_cols):
        # Fill between top_pattern_row and divider
        fill_rows = range(top_pattern_row+1, div_row)
    else:
        # Fill between divider and bot_pattern_row
        fill_rows = range(div_row+1, bot_pattern_row)
    
    for r in fill_rows:
        for c in intersection:
            out[r][c] = 4
    
    return out

def solve_78e78cff(grid):
    g = np.array(grid)
    H, W = g.shape
    out = g.copy()
    
    # Find the unique "seed" cell (not background, not border color)
    values = Counter(g.flatten())
    bg = values.most_common(1)[0][0]
    non_bg = [(r,c) for r in range(H) for c in range(W) if g[r][c] != bg]
    
    # Classify: border color and seed color
    border_colors = set()
    seed_cell = None
    seed_color = None
    
    # Border color = value that appears in lines/patterns (not a singleton)
    color_counts = Counter(g[r][c] for r,c in non_bg)
    # Seed = smallest count (usually 1)
    seed_color = min(color_counts, key=lambda x: (color_counts[x], x))
    seed_positions = [(r,c) for r,c in non_bg if g[r][c] == seed_color]
    border_color = [v for v in color_counts if v != seed_color][0]
    
    # Flood fill from seed position
    if seed_positions:
        sr, sc = seed_positions[0]
        # BFS flood fill blocked by border_color
        visited = set()
        q = deque([(sr, sc)])
        while q:
            r, c = q.popleft()
            if (r,c) in visited: continue
            visited.add((r,c))
            out[r][c] = seed_color
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<H and 0<=nc<W and (nr,nc) not in visited and g[nr][nc] != border_color:
                    q.append((nr, nc))
    
    return out.tolist()

def solve_780d0b14(grid):
    g = np.array(grid)
    H, W = g.shape
    
    # Find 0-rows and 0-columns that separate sections
    zero_rows = [r for r in range(H) if np.all(g[r,:] == 0)]
    zero_cols = [c for c in range(W) if np.all(g[:,c] == 0)]
    
    # Find row groups and col groups
    row_groups = []
    prev = -1
    for r in sorted(zero_rows + [H]):
        if r > prev+1:
            row_groups.append((prev+1, r))
        prev = r
    if not row_groups: row_groups = [(0, H)]
    
    col_groups = []
    prev = -1
    for c in sorted(zero_cols + [W]):
        if c > prev+1:
            col_groups.append((prev+1, c))
        prev = c
    if not col_groups: col_groups = [(0, W)]
    
    result = []
    for (r0, r1) in row_groups:
        row_result = []
        for (c0, c1) in col_groups:
            region = g[r0:r1, c0:c1]
            vals = [v for v in region.flatten() if v != 0]
            if vals:
                color = Counter(vals).most_common(1)[0][0]
                row_result.append(int(color))
            # else skip
        if row_result:
            result.append(row_result)
    return result

def solve_782b5218(grid):
    g = np.array(grid)
    H, W = g.shape
    out = g.copy()
    
    # Background color = most common, fill color = second most common non-2
    vals = Counter(g.flatten())
    # 2 is the barrier
    non_two = {v: c for v, c in vals.items() if v != 2}
    if not non_two: return grid
    
    fill_color = max(non_two, key=lambda v: non_two[v])
    
    # For each column, find the topmost 2
    for c in range(W):
        first_two_row = None
        for r in range(H):
            if g[r, c] == 2:
                first_two_row = r
                break
        if first_two_row is None: continue
        # Above the 2: set to 0
        for r in range(first_two_row):
            out[r, c] = 0
        # The 2 stays
        # Below the 2: set to fill_color
        for r in range(first_two_row+1, H):
            out[r, c] = fill_color
    
    return out.tolist()

def solve_7acdf6d3(grid):
    g = [list(r) for r in grid]
    H, W = len(g), len(g[0])
    out = [row[:] for row in g]
    
    # Find 2-shapes (V-shapes: 3 cells with one "gap" in the middle row)
    # Pattern: [2,_,2] in one row, [_,2,_] in next row = triangle
    # Or: [_,2,_] top, [2,_,2] bottom = another triangle
    
    # Remove all 9s from output first
    non_two_non_bg = None
    for r in range(H):
        for c in range(W):
            if g[r][c] not in [0,2,7]:  # assuming bg is 7
                if g[r][c] not in [2]:
                    non_two_non_bg = g[r][c]
    
    # Remove original 9s
    bg = Counter(g[r][c] for r in range(H) for c in range(W)).most_common(1)[0][0]
    for r in range(H):
        for c in range(W):
            if g[r][c] not in [0, bg, 2]:
                out[r][c] = bg
    
    # Find V-shapes of 2s and fill interior with 9
    # Pattern A: two 2s in row r, one 2 in row r+1 between them
    # Pattern B: one 2 in row r, two 2s in row r+1 on either side
    twos = {(r,c) for r in range(H) for c in range(W) if g[r][c]==2}
    
    # Find all 3-cell L-shapes that form a "bowl" or "inverted bowl"
    # Type 1: (r,c1), (r,c2), (r+1,(c1+c2)//2) where c2=c1+2
    for r in range(H-1):
        for c in range(W-2):
            if (r,c) in twos and (r,c+2) in twos and (r+1,c+1) in twos:
                # Bowl: fill interior = (r, c+1)
                out[r][c+1] = 9
    # Type 2: inverted bowl
    for r in range(1, H):
        for c in range(W-2):
            if (r,c) in twos and (r,c+2) in twos and (r-1,c+1) in twos:
                out[r][c+1] = 9
    
    # Also handle larger V-shapes (from 5-cell pattern)
    # Pattern: top row [2,0,0,2], middle rows [2,_,_,2], bottom [0,2,_,2,0]
    # Just flood fill? No - find enclosed regions within 2-boundaries
    
    return out

def solve_7b6016b9(grid):
    g = np.array(grid)
    H, W = g.shape
    out = np.full_like(g, 3)  # background -> 3
    
    # Copy non-zero cells (the 1s/8s framework)
    line_color = None
    for v in set(g.flatten()) - {0}:
        line_color = v; break
    
    for r in range(H):
        for c in range(W):
            if g[r,c] != 0:
                out[r,c] = g[r,c]
    
    # Find enclosed rectangular regions (bounded by line_color on all 4 sides)
    # A region is a connected component of 0s that is fully enclosed
    # BFS from border
    border_zeros = set()
    q = deque()
    for r in range(H):
        for c in [0, W-1]:
            if g[r,c] == 0:
                q.append((r,c)); border_zeros.add((r,c))
    for c in range(W):
        for r in [0, H-1]:
            if g[r,c] == 0 and (r,c) not in border_zeros:
                q.append((r,c)); border_zeros.add((r,c))
    
    while q:
        r,c = q.popleft()
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc = r+dr,c+dc
            if 0<=nr<H and 0<=nc<W and g[nr,nc]==0 and (nr,nc) not in border_zeros:
                border_zeros.add((nr,nc))
                q.append((nr,nc))
    
    # Interior zeros = enclosed regions -> fill with 2
    for r in range(H):
        for c in range(W):
            if g[r,c] == 0 and (r,c) not in border_zeros:
                out[r,c] = 2
    
    return out.tolist()

def solve_7c9b52a0(grid):
    g = np.array(grid)
    H, W = g.shape
    bg = int(Counter(g.flatten()).most_common(1)[0][0])
    
    # Find all rectangular "windows" (regions bounded by bg values)
    # Windows = connected regions of 0s within the bg
    non_bg = np.where(g != bg)
    if len(non_bg[0]) == 0: return grid
    
    # Find the "0-filled" rectangular windows
    # A window is a rectangle of 0s within the bg
    # Find connected components of non-bg cells
    from scipy.ndimage import label
    mask = (g != bg).astype(int)
    labeled, n = label(mask)
    
    windows = []
    for lbl in range(1, n+1):
        positions = list(zip(*np.where(labeled == lbl)))
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        r0, r1 = min(rows), max(rows)+1
        c0, c1 = min(cols), max(cols)+1
        windows.append((r0, r1, c0, c1, positions))
    
    # All windows should have the same size
    if not windows: return grid
    sizes = [(r1-r0, c1-c0) for r0,r1,c0,c1,_ in windows]
    target_size = Counter(sizes).most_common(1)[0][0]
    
    H_win, W_win = target_size
    result = [[0]*W_win for _ in range(H_win)]
    
    for r0, r1, c0, c1, positions in windows:
        if r1-r0 != H_win or c1-c0 != W_win: continue
        for r, c in positions:
            val = int(g[r, c])
            if val != bg and val != 0:
                result[r-r0][c-c0] = val
    
    return result

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

def solve_7ddcd7ec(grid):
    g = [list(r) for r in grid]
    H, W = len(g), len(g[0])
    out = [row[:] for row in g]
    
    # Find the solid block (NxN rectangle fully filled)
    non_zero = [(r,c,g[r][c]) for r in range(H) for c in range(W) if g[r][c] != 0]
    if not non_zero: return grid
    
    color = non_zero[0][2]
    positions = [(r,c) for r,c,v in non_zero]
    rows = [r for r,c in positions]
    cols = [c for r,c in positions]
    
    # Find the solid block (all cells in its bbox are filled)
    # Try to find a rectangular sub-region that's fully filled
    from scipy.ndimage import label
    arr = np.array([[1 if g[r][c]==color else 0 for c in range(W)] for r in range(H)])
    labeled, n = label(arr)
    
    # Find the block (largest connected component that's mostly a rectangle)
    block = None
    arms = []
    for lbl in range(1, n+1):
        pos = list(zip(*np.where(labeled==lbl)))
        rows_l = [p[0] for p in pos]
        cols_l = [p[1] for p in pos]
        r0, r1 = min(rows_l), max(rows_l)+1
        c0, c1 = min(cols_l), max(cols_l)+1
        area = len(pos)
        bbox_area = (r1-r0)*(c1-c0)
        if area == bbox_area and bbox_area > 1:
            block = (r0, r1, c0, c1)
        else:
            arms.extend(pos)
    
    if block is None: return grid
    
    br0, br1, bc0, bc1 = block
    
    # Extend each arm diagonally
    for (ar, ac) in arms:
        # Direction from nearest block corner to this arm cell
        # Find which corner of the block is closest
        # Actually: find direction from block corner through arm cell
        corners = [(br0-1, bc0-1), (br0-1, bc1), (br1, bc0-1), (br1, bc1)]
        # Find which corner the arm is extending from
        for cr, cc in corners:
            # Check if the arm is one diagonal step from this corner
            if abs(ar-cr) == 1 and abs(ac-cc) == 1:
                dr = ar - cr
                dc = ac - cc
                # Continue in this direction
                nr, nc = ar+dr, ac+dc
                while 0<=nr<H and 0<=nc<W:
                    out[nr][nc] = color
                    nr += dr; nc += dc
                break
    
    return out

def solve_7e4d4f7c(grid):
    g = grid
    row0 = g[0]
    row1 = g[1]
    
    # Background = most common in rows 1+
    from collections import Counter
    all_vals = [v for row in g[1:] for v in row]
    bg = Counter(all_vals).most_common(1)[0][0]
    
    # Background of row 0
    row0_bg = bg
    
    # col-0 values in rows 1+
    col0_vals = [g[r][0] for r in range(1, len(g)) if g[r][0] != bg]
    col0_marker = col0_vals[0] if col0_vals else 0
    
    # Build row 2
    row2 = []
    for c in range(len(row0)):
        v = row0[c]
        if col0_marker == 6:
            if v == row0_bg:
                row2.append(6)
            else:
                row2.append(v)
        else:
            if v != row0_bg:
                row2.append(6)
            else:
                row2.append(v)
    
    return [list(row0), list(row1), row2]

def solve_7ec998c9(grid):
    g = np.array(grid)
    H, W = g.shape
    bg = int(Counter(g.flatten()).most_common(1)[0][0])
    
    # Find the special cell
    special = [(r,c) for r in range(H) for c in range(W) if g[r][c] != bg]
    if not special: return grid
    sr, sc = special[0]
    special_val = int(g[sr, sc])
    
    out = g.copy().tolist()
    
    # Draw vertical line (1s at col sc, all rows except sr)
    for r in range(H):
        if r != sr:
            out[r][sc] = 1
    
    # At top edge (row 0): extend horizontally
    if sc >= W // 2:
        # Turn right
        for c in range(sc, W):
            out[0][c] = 1
    else:
        # Turn left
        for c in range(0, sc+1):
            out[0][c] = 1
    
    # At bottom edge (row H-1): opposite direction
    if sc >= W // 2:
        # Turn left
        for c in range(0, sc+1):
            out[H-1][c] = 1
    else:
        # Turn right
        for c in range(sc, W):
            out[H-1][c] = 1
    
    return out

def solve_7ee1c6ea(grid):
    g = [list(r) for r in grid]
    H, W = len(g), len(g[0])
    
    # Find the 5-border rectangle
    five_positions = [(r,c) for r in range(H) for c in range(W) if g[r][c]==5]
    if not five_positions: return grid
    
    rows = [p[0] for p in five_positions]
    cols = [p[1] for p in five_positions]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    
    # Find the two non-5, non-0 colors inside the border
    interior = [(r,c) for r in range(r0+1, r1) for c in range(c0+1, c1) if g[r][c] not in [0,5]]
    interior_colors = set(g[r][c] for r,c in interior)
    
    if len(interior_colors) < 2: return grid
    colors = list(interior_colors)
    a, b = colors[0], colors[1]
    
    # Swap a<->b inside the border
    out = [row[:] for row in g]
    for r in range(r0+1, r1):
        for c in range(c0+1, c1):
            if out[r][c] == a:
                out[r][c] = b
            elif out[r][c] == b:
                out[r][c] = a
    
    return out

def solve_7f4411dc(grid):
    g = np.array(grid)
    H, W = g.shape
    bg = 0
    out = np.zeros_like(g)
    
    # Find all connected components of non-zero cells
    from scipy.ndimage import label
    mask = (g != bg).astype(int)
    labeled, n = label(mask)
    
    for lbl in range(1, n+1):
        positions = list(zip(*np.where(labeled == lbl)))
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        r0, r1 = min(rows), max(rows)+1
        c0, c1 = min(cols), max(cols)+1
        area = len(positions)
        bbox_area = (r1-r0) * (c1-c0)
        
        # It's a solid rectangle if area == bbox_area and at least 2x2
        if area == bbox_area and (r1-r0) >= 2 and (c1-c0) >= 2:
            color = int(g[positions[0][0], positions[0][1]])
            out[r0:r1, c0:c1] = color
    
    return out.tolist()

