def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find all 7 positions (walls)
    walls = np.argwhere(grid == 7)
    
    # Find all non-7, non-6 positions (colored blocks)
    colored_mask = (grid != 7) & (grid != 6)
    colored_pos = np.argwhere(colored_mask)
    
    # Group colored blocks by value
    groups = {}
    for y, x in colored_pos:
        val = grid[y, x]
        groups.setdefault(val, []).append((y, x))
    
    # Find the 6 positions (floor)
    floor_pos = np.argwhere(grid == 6)
    if len(floor_pos) == 0:
        return grid.tolist()
    
    # Determine floor row (all 6s are in same row)
    floor_row = floor_pos[0, 0]
    
    # Process each colored group
    for val, positions in groups.items():
        # Skip if it's a single cell (likely a marker)
        if len(positions) <= 1:
            continue
        
        # Check if group forms a rectangle
        ys = [p[0] for p in positions]
        xs = [p[1] for p in positions]
        min_y, max_y = min(ys), max(ys)
        min_x, max_x = min(xs), max(xs)
        
        # Verify rectangle is filled
        rect_cells = [(y, x) for y in range(min_y, max_y+1) for x in range(min_x, max_x+1)]
        if set(rect_cells) == set(positions):
            # This is a rectangle group
            # Determine if it's above or below floor
            if max_y < floor_row:
                # Above floor: move down to floor, shift others right
                dy = floor_row - max_y - 1
                new_positions = [(y + dy, x + 1) for y, x in positions]
            else:
                # Below floor: move up to floor, shift others left
                dy = min_y - floor_row - 1
                new_positions = [(y - dy, x - 1) for y, x in positions]
            
            # Clear old positions
            for y, x in positions:
                grid[y, x] = 7
            
            # Place in new positions
            for (y, x), (ny, nx) in zip(positions, new_positions):
                if 0 <= ny < h and 0 <= nx < w:
                    grid[ny, nx] = val
    
    return grid.tolist()