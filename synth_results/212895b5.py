def transform(grid):
    import numpy as np
    
    grid = np.array(grid)
    h, w = grid.shape
    out = grid.copy()
    
    # Find all 5s and 8-blocks
    fives = list(zip(*np.where(grid == 5)))
    eights = list(zip(*np.where(grid == 8)))
    
    # Group 8s into blocks (connected components)
    eight_blocks = []
    visited = set()
    for pos in eights:
        if pos in visited:
            continue
        stack = [pos]
        block = []
        while stack:
            y, x = stack.pop()
            if (y, x) in visited:
                continue
            visited.add((y, x))
            block.append((y, x))
            for dy, dx in [(0,1),(0,-1),(1,0),(-1,0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 8:
                    stack.append((ny, nx))
        eight_blocks.append(block)
    
    # For each 5, find nearest 8-block
    for y5, x5 in fives:
        min_dist = float('inf')
        nearest_block = None
        for block in eight_blocks:
            # Use centroid of block
            by = sum(y for y, _ in block) / len(block)
            bx = sum(x for _, x in block) / len(block)
            dist = (y5 - by)**2 + (x5 - bx)**2
            if dist < min_dist:
                min_dist = dist
                nearest_block = block
        
        if nearest_block is None:
            continue
        
        # Find closest cell in that block
        min_cell_dist = float('inf')
        nearest_cell = None
        for y8, x8 in nearest_block:
            dist = abs(y5 - y8) + abs(x5 - x8)
            if dist < min_cell_dist:
                min_cell_dist = dist
                nearest_cell = (y8, x8)
        
        y8, x8 = nearest_cell
        
        # Draw 4-line from 5 to nearest 8-cell
        dy = y8 - y5
        dx = x8 - x5
        steps = max(abs(dy), abs(dx))
        if steps == 0:
            continue
        
        for i in range(1, steps + 1):
            y = y5 + (dy * i) // steps
            x = x5 + (dx * i) // steps
            if 0 <= y < h and 0 <= x < w and out[y, x] == 0:
                out[y, x] = 4
        
        # Draw 2-line from 8 to 5 (opposite direction)
        for i in range(1, steps + 1):
            y = y8 - (dy * i) // steps
            x = x8 - (dx * i) // steps
            if 0 <= y < h and 0 <= x < w and out[y, x] == 0:
                out[y, x] = 2
    
    return out.tolist()