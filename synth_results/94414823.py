def transform(grid):
    import numpy as np
    g = np.array(grid)
    R, C = g.shape
    box_color = 5
    
    # Find box
    box_rows = [r for r in range(R) if np.any(g[r] == box_color)]
    box_cols = [c for c in range(C) if np.any(g[:, c] == box_color)]
    r_top, r_bot = min(box_rows), max(box_rows)
    c_left, c_right = min(box_cols), max(box_cols)
    
    # Interior
    int_r1, int_r2 = r_top+1, r_bot-1
    int_c1, int_c2 = c_left+1, c_right-1
    int_h = int_r2 - int_r1 + 1
    int_w = int_c2 - int_c1 + 1
    
    # Find markers (non-zero, non-box cells outside the box)
    markers = []
    for r in range(R):
        for c in range(C):
            v = g[r, c]
            if v != 0 and v != box_color:
                # Determine relative position to box
                if r < r_top: side_r = 'top'
                elif r > r_bot: side_r = 'bottom'
                else: side_r = 'mid'
                if c < c_left: side_c = 'left'
                elif c > c_right: side_c = 'right'
                else: side_c = 'mid'
                markers.append((r, c, int(v), side_r, side_c))
    
    if len(markers) < 2:
        return grid
    
    # Assign quadrant colors based on marker positions
    # Quadrants: TL, TR, BL, BR
    quad_colors = {('top','left'): None, ('top','right'): None,
                   ('bottom','left'): None, ('bottom','right'): None}
    
    for _, _, v, sr, sc in markers:
        if sr in ('top','bottom') or sc in ('left','right'):
            key = (sr if sr != 'mid' else ('top' if True else 'bottom'),
                   sc if sc != 'mid' else ('left' if True else 'right'))
            # Use the dominant side
            qr = sr if sr != 'mid' else ('top' if _ < (r_top+r_bot)//2 else 'bottom')
            qc = sc if sc != 'mid' else ('left' if markers[0][1] < (c_left+c_right)//2 else 'right')
            quad_colors[(qr, qc)] = v
    
    # Fill in the other two quadrants (checkerboard)
    known = {k:v for k,v in quad_colors.items() if v is not None}
    if len(known) >= 2:
        items = list(known.items())
        (r1k,c1k),v1 = items[0]
        (r2k,c2k),v2 = items[1]
        # Diagonally opposite quadrants have same color
        opp = {'top':'bottom', 'bottom':'top', 'left':'right', 'right':'left'}
        quad_colors[(opp[r1k], opp[c1k])] = v1
        quad_colors[(opp[r2k], opp[c2k])] = v2
    
    # Fill interior
    out = g.copy()
    r_mid = int_r1 + int_h // 2
    c_mid = int_c1 + int_w // 2
    
    for r in range(int_r1, int_r2+1):
        for c in range(int_c1, int_c2+1):
            qr = 'top' if r < r_mid else 'bottom'
            qc = 'left' if c < c_mid else 'right'
            color = quad_colors.get((qr, qc))
            if color is not None:
                out[r, c] = color
    
    return out.tolist()
