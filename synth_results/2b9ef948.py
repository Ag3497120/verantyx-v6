import numpy as np
from collections import defaultdict

def find_components(arr, val, connectivity=4):
    rows, cols = arr.shape
    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if arr[r,c] == val and (r,c) not in visited:
                comp = []
                stack = [(r,c)]
                while stack:
                    rr,cc = stack.pop()
                    if (rr,cc) in visited or not (0<=rr<rows) or not (0<=cc<cols): continue
                    if arr[rr,cc] != val: continue
                    visited.add((rr,cc)); comp.append((rr,cc))
                    for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]: stack.append((rr+dr,cc+dc))
                if comp: components.append(comp)
    return components

def transform(grid):
    arr = np.array(grid)
    rows, cols = arr.shape
    
    # Find all non-zero, non-background values
    all_vals = set(arr.flatten().tolist()) - {0}
    
    # Find the rectangle (border of one color with center of another color)
    rect_color = None
    center_color = None
    rect_cells = []
    center_cell = None
    path_color = None  # 5s
    
    # Look for a 3x3 or similar rectangular border
    for v in all_vals:
        comps = find_components(arr, v)
        for comp in comps:
            rs = [r for r,c in comp]
            cs = [c for r,c in comp]
            min_r, max_r = min(rs), max(rs)
            min_c, max_c = min(cs), max(cs)
            h = max_r - min_r + 1
            w = max_c - min_c + 1
            # Check if it's a rectangular border
            expected_border = set()
            for rr in range(min_r, max_r+1):
                for cc in range(min_c, max_c+1):
                    if rr==min_r or rr==max_r or cc==min_c or cc==max_c:
                        expected_border.add((rr,cc))
            if set(comp) == expected_border and h >= 2 and w >= 2:
                rect_color = v
                rect_cells = comp
                # Find center value
                for rr in range(min_r+1, max_r):
                    for cc in range(min_c+1, max_c):
                        if arr[rr,cc] != 0 and arr[rr,cc] != v:
                            center_color = arr[rr,cc]
                            center_cell = (rr, cc)
    
    if rect_color is None:
        return grid
    
    # Find the path (5s)
    path_color_candidates = all_vals - {rect_color, center_color}
    for pv in path_color_candidates:
        pcs = find_components(arr, pv)
        if pcs:
            path_color = pv
            break
    
    # Find center_color marker outside rectangle
    rect_r = [r for r,c in rect_cells]
    rect_c = [c for r,c in rect_cells]
    rect_rmin, rect_rmax = min(rect_r), max(rect_r)
    rect_cmin, rect_cmax = min(rect_c), max(rect_c)
    
    center_marker = None
    border_marker = None
    for r in range(rows):
        for c in range(cols):
            v = arr[r,c]
            if v == center_color and not (rect_rmin<=r<=rect_rmax and rect_cmin<=c<=rect_cmax):
                center_marker = (r, c)
            elif v == rect_color and not (rect_rmin<=r<=rect_rmax and rect_cmin<=c<=rect_cmax):
                border_marker = (r, c)
    
    if center_marker is None or border_marker is None:
        return grid
    
    # Compute displacement from center_marker to border_marker
    dr = border_marker[0] - center_marker[0]
    dc = border_marker[1] - center_marker[1]
    
    # Rectangle dimensions
    h = rect_rmax - rect_rmin + 1
    w = rect_cmax - rect_cmin + 1
    
    # New rectangle position
    new_rmin = rect_rmin + dr
    new_rmax = rect_rmax + dr
    new_cmin = rect_cmin + dc
    new_cmax = rect_cmax + dc
    
    # Build output: fill background with center_color
    result = np.full((rows, cols), center_color, dtype=int)
    
    # Place new rectangle
    for rr in range(new_rmin, new_rmax+1):
        for cc in range(new_cmin, new_cmax+1):
            if 0<=rr<rows and 0<=cc<cols:
                if rr==new_rmin or rr==new_rmax or cc==new_cmin or cc==new_cmax:
                    result[rr,cc] = rect_color
                else:
                    result[rr,cc] = center_color
    
    # Draw X-diagonals from 4 corners
    corners = [(new_rmin, new_cmin), (new_rmin, new_cmax), (new_rmax, new_cmin), (new_rmax, new_cmax)]
    dir_map = {
        (new_rmin, new_cmin): [(-1,-1)],
        (new_rmin, new_cmax): [(-1,+1)],
        (new_rmax, new_cmin): [(+1,-1)],
        (new_rmax, new_cmax): [(+1,+1)],
    }
    for corner, dirs in dir_map.items():
        for d in dirs:
            r, c = corner[0]+d[0], corner[1]+d[1]
            while 0<=r<rows and 0<=c<cols:
                result[r,c] = rect_color
                r += d[0]; c += d[1]
    
    return result.tolist()
