def transform(grid):
    rows, cols = len(grid), len(grid[0])
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find the "frame" color - the most common non-background color among cells
    # that form a connected border
    # Strategy: find the dominant border color
    # The frame forms an irregular closed loop
    # Find the bounding box of the frame
    
    # Count all non-bg colors
    non_bg = [v for v in flat if v != bg]
    if not non_bg:
        return grid
    
    # The frame color is likely the color appearing in the border of the main structure
    color_count = Counter(non_bg)
    
    # Find bounding box of the main frame structure
    # Look for a closed rectangular loop shape
    # The frame cells form a connected ring
    from collections import deque
    
    # Find the main frame: the largest connected component of any non-bg, non-scattered cells
    # Actually: find all distinct colors and find which forms the frame
    # Simple approach: for each non-bg color, find its connected components
    # The frame color has a large component that forms a closed loop
    
    # Find the non-bg color with the most cells in a single connected component
    best_color = None
    best_comp = []
    
    for color in color_count:
        vis = set()
        comps = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color and (r,c) not in vis:
                    comp = []
                    q = deque([(r,c)])
                    vis.add((r,c))
                    while q:
                        cr,cc = q.popleft()
                        comp.append((cr,cc))
                        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr,nc = cr+dr, cc+dc
                            if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in vis and grid[nr][nc]==color:
                                vis.add((nr,nc))
                                q.append((nr,nc))
                    comps.append(comp)
        if comps:
            largest = max(comps, key=len)
            if len(largest) > len(best_comp):
                best_comp = largest
                best_color = color
    
    if not best_comp:
        return grid
    
    # Get bounding box of the frame
    rs = [r for r,c in best_comp]
    cs = [c for r,c in best_comp]
    r_min, r_max = min(rs), max(rs)
    c_min, c_max = min(cs), max(cs)
    
    # Extract this region
    result = []
    for r in range(r_min, r_max+1):
        row = []
        for c in range(c_min, c_max+1):
            row.append(grid[r][c])
        result.append(row)
    
    return result