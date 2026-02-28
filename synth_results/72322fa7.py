def transform(grid):
    return _solve(grid)

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


_solve = solve_72322fa7
