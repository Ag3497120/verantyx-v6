def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    out = grid.copy()
    
    # Find all non-zero single cells (potential sources)
    sources = []
    for y in range(h):
        for x in range(w):
            if grid[y, x] != 0:
                # Check if it's isolated (all 8 neighbors are 0)
                isolated = True
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if grid[ny, nx] != 0:
                                isolated = False
                if isolated:
                    sources.append((y, x, grid[y, x]))
    
    # For each source, look for the nearest 3x3 block of same non-zero value
    for sy, sx, val in sources:
        # Find all 3x3 blocks with uniform value > 0
        blocks = []
        for y in range(h - 2):
            for x in range(w - 2):
                block = grid[y:y+3, x:x+3]
                if np.all(block == block[0, 0]) and block[0, 0] > 0:
                    blocks.append((y, x, block[0, 0]))
        
        # Find the block with same value as source that is closest
        target_block = None
        min_dist = float('inf')
        for by, bx, bval in blocks:
            if bval == val:
                # Manhattan distance from source to block center
                dist = abs(sy - (by + 1)) + abs(sx - (bx + 1))
                if dist < min_dist:
                    min_dist = dist
                    target_block = (by, bx)
        
        if target_block:
            by, bx = target_block
            # Copy source value to block's top-left corner
            out[by, bx] = val
            # Copy block's bottom-right value to source position
            out[sy, sx] = grid[by + 2, bx + 2]
    
    return out.tolist()