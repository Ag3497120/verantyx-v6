def transform(grid):
    import numpy as np
    
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find all distinct colors that are not the background
    colors = np.unique(grid)
    # Background is the most frequent color (usually borders)
    bg_color = np.bincount(grid.flatten()).argmax()
    colors = [c for c in colors if c != bg_color]
    
    # Find connected components for each color
    from collections import defaultdict
    components = defaultdict(list)
    
    for color in colors:
        visited = np.zeros((h, w), dtype=bool)
        for y in range(h):
            for x in range(w):
                if grid[y, x] == color and not visited[y, x]:
                    # BFS to find component
                    stack = [(y, x)]
                    comp = []
                    while stack:
                        cy, cx = stack.pop()
                        if visited[cy, cx]:
                            continue
                        visited[cy, cx] = True
                        comp.append((cy, cx))
                        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if grid[ny, nx] == color and not visited[ny, nx]:
                                    stack.append((ny, nx))
                    if comp:
                        components[color].append(comp)
    
    # Find the largest component for each color (likely the shape interior)
    largest_comps = {}
    for color in components:
        largest = max(components[color], key=len)
        largest_comps[color] = largest
    
    # Extract bounding box for each largest component
    bboxes = {}
    for color in largest_comps:
        comp = largest_comps[color]
        ys = [p[0] for p in comp]
        xs = [p[1] for p in comp]
        ymin, ymax = min(ys), max(ys)
        xmin, xmax = min(xs), max(xs)
        bboxes[color] = (ymin, ymax, xmin, xmax)
    
    # Find the union of all bboxes (the overall region containing all shapes)
    all_ymin = min(bb[0] for bb in bboxes.values())
    all_ymax = max(bb[1] for bb in bboxes.values())
    all_xmin = min(bb[2] for bb in bboxes.values())
    all_xmax = max(bb[3] for bb in bboxes.values())
    
    # Crop to that region
    cropped = grid[all_ymin:all_ymax+1, all_xmin:all_xmax+1]
    
    # Replace background with 0 for easier processing
    cropped_bg = np.bincount(cropped.flatten()).argmax()
    result = cropped.copy()
    result[result == cropped_bg] = 0
    
    # Find the minimal bounding box that contains all non-zero pixels
    nonzeros = np.where(result != 0)
    if len(nonzeros[0]) == 0:
        return []
    ymin, ymax = nonzeros[0].min(), nonzeros[0].max()
    xmin, xmax = nonzeros[1].min(), nonzeros[1].max()
    final = result[ymin:ymax+1, xmin:xmax+1]
    
    return final.tolist()