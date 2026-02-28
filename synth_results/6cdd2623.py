def transform(grid):
    H = len(grid)
    W = len(grid[0])
    result = [[0] * W for _ in range(H)]
    
    # Find color that only appears on grid borders and forms a cross
    from collections import defaultdict
    color_positions = defaultdict(list)
    all_colors = set()
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v != 0:
                all_colors.add(v)
                color_positions[v].append((r, c))
    
    # Find color whose ALL positions are on the border
    for color, positions in color_positions.items():
        all_on_border = all(r == 0 or r == H-1 or c == 0 or c == W-1 for r, c in positions)
        if not all_on_border:
            continue
        # Find rows with color at both left and right
        rows_filled = set()
        for r, c in positions:
            if c == 0 and any(r2 == r and c2 == W-1 for r2, c2 in positions):
                rows_filled.add(r)
        # Find cols with color at both top and bottom
        cols_filled = set()
        for r, c in positions:
            if r == 0 and any(r2 == H-1 and c2 == c for r2, c2 in positions):
                cols_filled.add(c)
        
        if rows_filled or cols_filled:
            for r in rows_filled:
                for c in range(W):
                    result[r][c] = color
            for col in cols_filled:
                for r in range(H):
                    if result[r][col] == 0:
                        result[r][col] = color
            break
    
    return result
