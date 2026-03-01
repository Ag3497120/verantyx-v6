def transform(grid):
    rows, cols = len(grid), len(grid[0])
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    colors = set(flat) - {bg}
    
    # Identify frame colors vs noise
    frame_colors = []
    for color in colors:
        cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == color]
        if len(cells) < 4:
            continue
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)
        width = max_c - min_c + 1
        height = max_r - min_r + 1
        # Frame: top and bottom rows are mostly filled
        top_count = sum(1 for c2 in range(min_c, max_c + 1) if grid[min_r][c2] == color)
        bot_count = sum(1 for c2 in range(min_c, max_c + 1) if grid[max_r][c2] == color)
        if top_count >= width * 0.5 and bot_count >= width * 0.5 and height >= 3 and width >= 3:
            frame_colors.append(color)
    
    all_frame = set(frame_colors)
    noise_candidates = colors - all_frame
    noise_color = None
    if noise_candidates:
        noise_color = max(noise_candidates, key=lambda c: sum(1 for v in flat if v == c))
    else:
        noise_color = bg  # fallback
    
    # For each frame, count regions using partition analysis
    frame_info = []
    
    for color in frame_colors:
        cells = set((r, c) for r in range(rows) for c in range(cols) if grid[r][c] == color)
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)
        
        # Count vertical partitions: columns where all interior cells are non-bg
        v_parts = 0
        for c2 in range(min_c + 1, max_c):
            all_nonbg = True
            for r2 in range(min_r + 1, max_r):
                if grid[r2][c2] == bg:
                    all_nonbg = False
                    break
            if all_nonbg:
                v_parts += 1
        
        # Count horizontal partitions
        h_parts = 0
        for r2 in range(min_r + 1, max_r):
            all_nonbg = True
            for c2 in range(min_c + 1, max_c):
                if grid[r2][c2] == bg:
                    all_nonbg = False
                    break
            if all_nonbg:
                h_parts += 1
        
        region_count = (v_parts + 1) * (h_parts + 1)
        frame_info.append((color, region_count))
    
    # Sort by region count ascending
    frame_info.sort(key=lambda x: x[1])
    
    max_regions = max(r for _, r in frame_info) if frame_info else 1
    out = []
    for color, count in frame_info:
        row = [color] * count + [noise_color] * (max_regions - count)
        out.append(row)
    
    return out
