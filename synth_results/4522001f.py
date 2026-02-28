def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    # Find the special value (2 = scale factor indicator)
    # Each non-zero, non-2 cell becomes a block scaled by 2's position
    # Actually: the grid has cells with 0 (background) and 3 (foreground), and 2 (anchor)
    # Scale = 3^(something)? Let me look at examples:
    # Input 3x3, one 2, rest 0 and 3. Output is 9x9.
    # The 3s in the input → 4x4 block of 3 in output?
    
    # Find position of 2 (scale indicator?)
    # Actually: scale factor = 3 (hard-coded based on output size / input size)
    # Let me check: input is 3x3, output is 9x9 = 3^2 scale? No 9/3=3 = scale of 3.
    
    # Find background color (most common)
    from collections import Counter
    cnt = Counter(v for row in grid for v in row)
    bg = cnt.most_common(1)[0][0]
    
    # Find the scale factor - it's the size of the non-bg region?
    # Try: scale = 3 if input is 3x3, or look for the 2
    two_pos = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 2]
    
    # Scale = number of non-zero values? Or find from pattern
    # Let's try: scale factor = position of 2 + 1? No.
    # Actually: look at unique non-zero values
    vals = set(v for row in grid for v in row if v != 0)
    
    scale = 3  # default
    
    if 2 in vals:
        # Each cell maps to scale x scale block
        # The 2 maps to... 0 in output? Let's assume it means "scale indicator"
        # And the actual content color is the other value
        content_color = [v for v in vals if v not in (0, 2)][0] if len(vals) > 1 else list(vals)[0]
        
        # Build output: each cell becomes scale x scale block
        out_rows = rows * scale
        out_cols = cols * scale
        result = [[0]*out_cols for _ in range(out_rows)]
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == content_color:
                    for dr in range(scale):
                        for dc in range(scale):
                            result[r*scale+dr][c*scale+dc] = content_color
        
        return result
    
    return [list(row) for row in grid]
