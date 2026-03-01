def transform(grid):
    rows = len(grid); cols = len(grid[0])
    from collections import Counter, deque
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find all rectangular "rings" (frames) and their interiors
    # A ring: connected component of same color forming a rectangular border
    visited = [[False]*cols for _ in range(rows)]
    
    def bfs_comp(sr, sc, val):
        comp = []; stack = [(sr,sc)]
        while stack:
            r,c = stack.pop()
            if r<0 or r>=rows or c<0 or c>=cols: continue
            if visited[r][c] or grid[r][c]!=val: continue
            visited[r][c]=True; comp.append((r,c))
            for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]: stack.append((r+dr,c+dc))
        return comp
    
    rings = {}  # color -> {interior_val: V, r0, r1, c0, c1}
    result = [row[:] for row in grid]
    
    comps = []
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != bg:
                val = grid[r][c]
                comp = bfs_comp(r, c, val)
                rs = [x for x,y in comp]; cs2 = [y for x,y in comp]
                r0,r1 = min(rs),max(rs); c0,c1 = min(cs2),max(cs2)
                # Find interior non-bg cells
                interior_vals = []
                for ir in range(r0+1, r1):
                    for ic in range(c0+1, c1):
                        if grid[ir][ic] != bg: interior_vals.append((ir,ic,grid[ir][ic]))
                if interior_vals:
                    # It's a ring
                    interior_val = interior_vals[0][2] if len(interior_vals)==1 else Counter([v for _,_,v in interior_vals]).most_common(1)[0][0]
                    comps.append({'color': val, 'comp': comp, 'r0':r0,'r1':r1,'c0':c0,'c1':c1,
                                  'interior_val': interior_val, 'interior_cells': [(ir,ic) for ir,ic,_ in interior_vals]})
    
    # Build directed graph: ring color -> ring interior_val
    ring_map = {c['color']: c for c in comps}
    
    # Topological sort processing
    in_degree = {c['color']: 0 for c in comps}
    for c in comps:
        iv = c['interior_val']
        if iv in ring_map:
            in_degree[iv] += 1
    
    # Process in topological order
    # roots = rings with in_degree 0
    consumed = set()
    new_interiors = {}  # color -> new interior value
    
    remaining = {c['color'] for c in comps}
    
    while remaining:
        roots = [col for col in remaining if in_degree.get(col,0) == 0]
        if not roots:
            # Cycle - break it
            roots = [next(iter(remaining))]
        
        for root_color in roots:
            if root_color not in remaining: continue
            ring = ring_map[root_color]
            iv = ring['interior_val']
            if iv in ring_map and iv not in consumed:
                # Get iv's interior
                new_interiors[root_color] = ring_map[iv]['interior_val']
                # Mark iv as consumed
                consumed.add(iv)
                remaining.discard(iv)
                # Decrease in_degree of ring that iv would point to
                next_iv = ring_map[iv]['interior_val']
                if next_iv in ring_map:
                    in_degree[next_iv] = max(0, in_degree.get(next_iv,0)-1)
            else:
                new_interiors[root_color] = bg
            remaining.discard(root_color)
            if iv in ring_map and iv not in consumed:
                pass  # iv was already processed or consumed
    
    # Apply new interiors
    for c in comps:
        color = c['color']
        new_val = new_interiors.get(color, bg)
        for ir, ic in c['interior_cells']:
            result[ir][ic] = new_val
    
    # Remove consumed rings (clear their cells to bg)
    for color in consumed:
        for r,c2 in ring_map[color]['comp']:
            result[r][c2] = bg
        for ir, ic in ring_map[color]['interior_cells']:
            result[ir][ic] = bg
    
    return result
