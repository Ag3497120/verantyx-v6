
def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    result = g.copy()
    
    # Find the box (surrounded by different color border)
    # The box is a 3x3 bordered region
    # Find all distinct non-zero, non-background values
    from collections import Counter
    cnt = Counter(g.flatten().tolist())
    if 0 in cnt: del cnt[0]
    
    # Find cells of color 2 (the border)
    # The box has color 2 forming a 3x3 border around a center
    # Find isolated 3s and the box
    colors = [c for c in cnt]
    # Find the box: a 3x3 square of one color (the border color)
    box_color = None
    box_pos = None
    for c in colors:
        cells = list(zip(*np.where(g == c)))
        if len(cells) > 1:
            rs = [r for r,cc in cells]
            cs = [cc for r,cc in cells]
            if max(rs)-min(rs) <= 2 and max(cs)-min(cs) <= 2:
                # Could be box border
                box_color = c
                box_r0, box_c0 = min(rs), min(cs)
                box_pos = (box_r0, box_c0)
                break
    
    if box_pos is None:
        return grid
    
    br0, bc0 = box_pos
    # Find the center of the box
    center_color = None
    for dr in range(3):
        for dc in range(3):
            if g[br0+dr, bc0+dc] != box_color and g[br0+dr, bc0+dc] != 0:
                center_color = g[br0+dr, bc0+dc]
    
    # Find isolated markers (same as center color but isolated)
    if center_color is None:
        center_color = 3  # default
    
    # Find nearest marker outside the box
    markers = [(r,c) for r,c in zip(*np.where(g == center_color))
               if not (br0 <= r <= br0+2 and bc0 <= c <= bc0+2)]
    
    if not markers:
        return grid
    
    # Find nearest marker
    box_center = (br0+1, bc0+1)
    nearest = min(markers, key=lambda m: abs(m[0]-box_center[0])+abs(m[1]-box_center[1]))
    
    # Move box to nearest marker position
    # Determine direction (horizontal or vertical)
    nr, nc = nearest
    
    # Erase box
    result[br0:br0+3, bc0:bc0+3] = 0
    # Place marker where box was
    result[box_center[0], box_center[1]] = center_color
    
    # Place box centered on nearest
    new_br0, new_bc0 = nr-1, nc-1
    for dr in range(3):
        for dc in range(3):
            rr, cc = new_br0+dr, new_bc0+dc
            if 0 <= rr < rows and 0 <= cc < cols:
                if dr==1 and dc==1:
                    result[rr, cc] = center_color
                else:
                    result[rr, cc] = box_color
    # Remove the nearest marker (it's now inside the box)
    # The marker position becomes center of new box
    
    return result.tolist()
