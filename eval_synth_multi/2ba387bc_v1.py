def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    
    # Find all 4x4 colored rectangles
    visited = set()
    rects = []
    
    for r in range(rows - 3):
        for c in range(cols - 3):
            if (r, c) in visited or g[r, c] == 0:
                continue
            # Check if this is top-left of a 4x4 rect
            block = g[r:r+4, c:c+4]
            colors = set(block.flatten()) - {0}
            if len(colors) == 1:
                color = colors.pop()
                # Check it's actually isolated (4x4 block)
                # Classify as hollow or solid
                inner = block[1:3, 1:3]
                if np.all(inner == 0):
                    rects.append((r, c, color, 'hollow', block.copy()))
                elif np.all(block == color):
                    rects.append((r, c, color, 'solid', block.copy()))
                else:
                    continue
                for dr in range(4):
                    for dc in range(4):
                        visited.add((r+dr, c+dc))
    
    # Sort by reading order
    hollow = sorted([x for x in rects if x[3] == 'hollow'], key=lambda x: (x[0], x[1]))
    solid = sorted([x for x in rects if x[3] == 'solid'], key=lambda x: (x[0], x[1]))
    
    n = max(len(hollow), len(solid))
    result = []
    for i in range(n):
        h_block = hollow[i][4] if i < len(hollow) else np.zeros((4, 4), dtype=int)
        s_block = solid[i][4] if i < len(solid) else np.zeros((4, 4), dtype=int)
        combined = np.hstack([h_block, s_block])
        result.extend(combined.tolist())
    
    return result
