def transform(grid):
    import numpy as np
    from collections import Counter
    
    g = np.array(grid)
    h, w = g.shape
    
    # Background color (most common non-1 value)
    counts = Counter(g.flatten().tolist())
    counts.pop(1, None)
    if not counts:
        return grid
    bg = counts.most_common(1)[0][0]
    
    result = g.copy()
    
    # Find connected components of 1s
    visited = np.zeros((h, w), dtype=bool)
    
    def find_component(r0, c0):
        comp = set()
        stack = [(r0, c0)]
        while stack:
            r, c = stack.pop()
            if (r, c) in comp:
                continue
            comp.add((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<h and 0<=nc<w and g[nr,nc]==1 and (nr,nc) not in comp:
                    stack.append((nr, nc))
        return comp
    
    components = []
    for r in range(h):
        for c in range(w):
            if g[r,c] == 1 and not visited[r,c]:
                comp = find_component(r, c)
                for pr, pc in comp:
                    visited[pr, pc] = True
                components.append(comp)
    
    for comp in components:
        rows = [r for r, c in comp]
        cols = [c for r, c in comp]
        r_min, r_max = min(rows), max(rows)
        c_min, c_max = min(cols), max(cols)
        
        if r_max <= r_min or c_max <= c_min:
            continue
        
        # Fill interior with 2
        for r in range(r_min + 1, r_max):
            for c in range(c_min + 1, c_max):
                if result[r, c] == bg:
                    result[r, c] = 2
        
        # Find gaps on each side and propagate outward
        # Top side (row r_min)
        for c in range(c_min, c_max + 1):
            if (r_min, c) not in comp and result[r_min, c] == bg:
                result[r_min, c] = 2
                r = r_min - 1
                while r >= 0 and result[r, c] == bg:
                    result[r, c] = 2
                    r -= 1
        
        # Bottom side (row r_max)
        for c in range(c_min, c_max + 1):
            if (r_max, c) not in comp and result[r_max, c] == bg:
                result[r_max, c] = 2
                r = r_max + 1
                while r < h and result[r, c] == bg:
                    result[r, c] = 2
                    r += 1
        
        # Left side (col c_min)
        for r in range(r_min + 1, r_max):
            if (r, c_min) not in comp and result[r, c_min] == bg:
                result[r, c_min] = 2
                c = c_min - 1
                while c >= 0 and result[r, c] == bg:
                    result[r, c] = 2
                    c -= 1
        
        # Right side (col c_max)
        for r in range(r_min + 1, r_max):
            if (r, c_max) not in comp and result[r, c_max] == bg:
                result[r, c_max] = 2
                c = c_max + 1
                while c < w and result[r, c] == bg:
                    result[r, c] = 2
                    c += 1
    
    return result.tolist()
