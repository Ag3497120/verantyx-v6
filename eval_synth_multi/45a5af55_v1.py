def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    
    # Extract horizontal bands
    bands = []
    r = 0
    while r < rows:
        color = int(g[r, 0])
        thickness = 0
        while r + thickness < rows and int(g[r + thickness, 0]) == color:
            thickness += 1
        bands.append((color, thickness))
        r += thickness
    
    # Compute output size
    total = sum(t for _, t in bands)
    out_size = 2 * total - 2 * bands[-1][1]  # innermost doesn't double
    
    # Actually: the output is a square. The side length = 2 * sum of all thicknesses except we don't double the innermost
    # Wait, let me recalculate: each band adds its thickness to all 4 sides of the frame
    # Total side = 2 * (sum of all thicknesses except innermost) + innermost_thickness * 2? No...
    # Actually output side = 2 * sum(all thicknesses)
    # For Train 0: sum = 2+5+1+2+2+2 = 14. Output = 26 = 2*14 - 2. 
    # The -2 is because... hmm, 2*14=28 but output is 26.
    
    # Let me re-examine: input cols = 14. Output side = 26. 
    # Each band wraps around. The width that each band occupies on left+right = 2*thickness
    # But the innermost band: its height in output center = thickness*2? No, 2 rows → 2 in center.
    # Total side = 2 * sum(all thicknesses). For Train 0: 2*14 = 28. But output is 26.
    
    # Hmm, maybe the input width matters. Let me check: input is 14x14, output 26x26.
    # Input width = 14. Width contribution of each band on left and right sides.
    # The innermost band fills a rectangle of size (width - extra) * (thickness*2).
    # Actually maybe the "width" of each band in the output is the input column count.
    
    # Let me look at it differently. The output has nested rectangles.
    # The outermost band (8, thickness 2): occupies top 2 rows, bottom 2 rows, left 2 cols, right 2 cols.
    # The next band (2, thickness 5): inside the 8-frame. Occupies 5 cells on each side.
    # The next (6, thickness 1): 1 cell on each side.
    # The next (8, thickness 2): 2 cells on each side.
    # The next (1, thickness 2): 2 cells on each side.
    # The innermost (2, thickness 2): fills the center rectangle.
    
    # Output size = 2 * sum(all thicknesses) = 2 * 14 = 28? But output is 26.
    # 26 = 14 + 12? Hmm...
    # Actually output cols = 26. Let me trace: from outside in, each frame adds thickness to each side.
    # Total width = 2*(t1 + t2 + ... + t_{n-1}) + innermost_width
    # The innermost has both its top/bottom thickness AND width from input cols.
    # Wait... input has cols=14, each row is uniform color. The "bands" run horizontally across all 14 cols.
    # The output wraps these bands into a square frame structure.
    # Maybe the output width = input_cols + 2*(sum of thicknesses - innermost_thickness)?
    # = 14 + 2*(14-2) = 14 + 24 = 38. No.
    
    # Let me just measure: output row 12 (inside all frames): 
    # [8,8,2,2,2,2,2,6,8,8,1,1,2,2,1,1,8,8,6,2,2,2,2,2,8,8]
    # The innermost 2 occupies cols 12-13 (2 wide). 
    # Then 1 occupies cols 10-11 and 14-15 (2 thick each side). Inner rect: 6 wide.
    # Then 8 at cols 8-9 and 16-17 (2 thick). Inner rect: 10 wide.
    # Then 6 at col 7 and 18 (1 thick). Inner rect: 12 wide.
    # Then 2 at cols 2-6 and 19-23 (5 thick). Inner rect: 22 wide.
    # Then 8 at cols 0-1 and 24-25 (2 thick). Full width: 26.
    
    # So output width = 2 * (2 + 5 + 1 + 2 + 2) + 2 = 2*12 + 2 = 26.
    # = 2 * (sum - innermost_thickness) + 2 * innermost_thickness? That's 2*sum = 28. No.
    # Wait: 2 + 2*5 + 2*1 + 2*2 + 2*2 + 2 = 2+10+2+4+4+2 = 24. No.
    # Hmm: innermost is 2 wide (thickness 2). 
    # Then each frame adds 2*thickness to the total.
    # 2 (innermost) + 2*2 (1) + 2*2 (8) + 2*1 (6) + 2*5 (2) + 2*2 (8) = 2+4+4+2+10+4 = 26. ✓
    
    side = sum(2 * t for _, t in bands)
    # Wait that's 2*14=28. But we calculated 26. Because innermost counts once, not twice?
    # Actually: 2+4+4+2+10+4 = 26. But 2*2 + 2*5 + 2*1 + 2*2 + 2*2 + 2*2 = 28.
    # The innermost is 2*2 = 4, but we counted 2. So innermost counts as thickness, not 2*thickness.
    # Wait no: the innermost rectangle has width = 2*last_thickness and height = 2*last_thickness? No...
    
    # Let me re-count: innermost band (2, thickness 2). In output center: rows 12-13, cols 12-13. That's 2x2.
    # Next band (1, thickness 2): adds 2 on each side. New rect: (10,10) to (15,15) = 6x6. The 1 occupies the frame.
    # Next band (8, thickness 2): adds 2 on each side. New rect: (8,8) to (17,17) = 10x10.
    # Next band (6, thickness 1): adds 1 on each side. New rect: (7,7) to (18,18) = 12x12.
    # Next band (2, thickness 5): adds 5 on each side. New rect: (2,2) to (23,23) = 22x22.
    # Next band (8, thickness 2): adds 2 on each side. New rect: (0,0) to (25,25) = 26x26.
    
    # Inner size for the last band: input_cols? No, it's just 2*thickness of innermost.
    # Actually the innermost rectangle has width=height= just the cell count needed.
    # The center block has width = cols (14)? No, it's 2x2.
    
    # So the center is NOT related to input cols. The center = 2*innermost_thickness? No, 2*2=4 but center is 2x2.
    # Center = innermost_thickness x innermost_thickness? 2x2 = yes!
    # Wait, let me check: center block rows 12-13, cols 12-13. That's 2 rows x 2 cols = thickness x thickness.
    # Hmm, but thickness is the number of rows of that color in the input, not a side length.
    
    # Actually I think the center is thickness_last (height) x cols? But cols=14 and center is 2 wide.
    # 
    # I think the pattern is: the bands represent CONCENTRIC FRAMES. Each band's thickness defines 
    # how thick the frame is. Going from outermost to innermost.
    # The TOTAL size of the square = sum of all contributions from each side.
    # Each frame adds its thickness on each side (left, right, top, bottom).
    # But for height, the innermost contributes its thickness (not doubled since it's the center fill).
    # For width, similarly.
    # Total side = 2 * sum(thicknesses[:-1]) + thicknesses[-1]? 
    # = 2*(2+5+1+2+2) + 2 = 24 + 2 = 26. ✓
    
    n = len(bands)
    side = 2 * sum(t for _, t in bands[:-1]) + bands[-1][1]
    # Wait: that gives 2*12 + 2 = 26. But the center is 2x2, not 2x1.
    # Center is square: thickness x thickness. And we're computing side length.
    # With n bands, frame i (0-indexed from outside) adds thickness[i] on each side.
    # Inner dimension after all frames = side - 2*sum(thickness[0:n-1]) = thickness[n-1].
    # But for a square, the inner-most frame fills a thickness[n-1] x thickness[n-1] square.
    # That means the inner dimension = thickness[n-1]. 
    # side = 2*sum(thickness[0:n-1]) + thickness[n-1]. ✓ for Train 0.
    
    # But wait, for Train 1 let me check: input 13x12.
    # I need to verify with Train 1 too. Let me just code it and test.
    
    out_side = 2 * sum(t for _, t in bands[:-1]) + bands[-1][1]
    
    out = np.full((out_side, out_side), 0, dtype=int)
    
    offset = 0
    for i, (color, thickness) in enumerate(bands):
        if i == n - 1:
            # Fill center
            out[offset:offset+thickness, offset:offset+thickness] = color
        else:
            # Fill frame
            end = out_side - offset
            # Top
            out[offset:offset+thickness, offset:end] = color
            # Bottom
            out[end-thickness:end, offset:end] = color
            # Left
            out[offset:end, offset:offset+thickness] = color
            # Right
            out[offset:end, end-thickness:end] = color
            offset += thickness
    
    return out.tolist()
