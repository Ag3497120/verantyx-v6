
def transform(grid):
    from collections import Counter, deque
    R, C = len(grid), len(grid[0])
    cnt = Counter(grid[r][c] for r in range(R) for c in range(C))
    bg = cnt.most_common(1)[0][0]
    out = [row[:] for row in grid]
    # Find closed regions: shapes that are NOT closed have open sides
    # The shape cells and interior define a region
    # Check which sides of the bounding box are open (missing border cells)
    # For each distinct shape color, find its bounding box and open sides
    # Then emit rays in those directions from the shape cells on that side
    
    nz_colors = set(grid[r][c] for r in range(R) for c in range(C)) - {bg}
    for color in nz_colors:
        cells = [(r,c) for r in range(R) for c in range(C) if grid[r][c]==color]
        if not cells: continue
        min_r = min(r for r,c in cells)
        max_r = max(r for r,c in cells)
        min_c = min(c for r,c in cells)
        max_c = max(c for r,c in cells)
        
        # Check which sides are open (don't have any cells)
        top_cells = [(r,c) for r,c in cells if r==min_r]
        bot_cells = [(r,c) for r,c in cells if r==max_r]
        left_cells = [(r,c) for r,c in cells if c==min_c]
        right_cells = [(r,c) for r,c in cells if c==max_c]
        
        top_cols = set(c for r,c in top_cells)
        bot_cols = set(c for r,c in bot_cells)
        left_rows = set(r for r,c in left_cells)
        right_rows = set(r for r,c in right_cells)
        
        all_cols = set(range(min_c, max_c+1))
        all_rows = set(range(min_r, max_r+1))
        
        # Find interior color by flood fill from outside
        # The fill = most common non-bg, non-border color inside
        # Actually: the interior cells are whatever is inside the shape outline
        # For rays: look for which side is "open" (missing border cells)
        
        # Check missing sides
        missing_top = all_cols - top_cols
        missing_bot = all_cols - bot_cols
        missing_left = all_rows - left_rows
        missing_right = all_rows - right_rows
        
        # Find interior color
        interior_cells = [(r,c) for r in range(min_r+1, max_r) for c in range(min_c+1, max_c)
                         if grid[r][c] != bg and grid[r][c] != color]
        if not interior_cells:
            # Try flood fill
            interior_color = color
        else:
            interior_color = Counter(grid[r][c] for r,c in interior_cells).most_common(1)[0][0]
        
        # Emit rays from open sides
        if missing_top:
            # Open on top: emit rays upward from top cells
            for r,c in top_cells:
                for nr in range(r-1, -1, -1):
                    if grid[nr][c] == bg:
                        out[nr][c] = interior_color
                    else:
                        break
        if missing_bot:
            for r,c in bot_cells:
                for nr in range(r+1, R):
                    if grid[nr][c] == bg:
                        out[nr][c] = interior_color
                    else:
                        break
        if missing_left:
            for r,c in left_cells:
                for nc in range(c-1, -1, -1):
                    if grid[r][nc] == bg:
                        out[r][nc] = interior_color
                    else:
                        break
        if missing_right:
            for r,c in right_cells:
                for nc in range(c+1, C):
                    if grid[r][nc] == bg:
                        out[r][nc] = interior_color
                    else:
                        break
    return out
