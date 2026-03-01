def transform(grid_list):
    import numpy as np
    from collections import Counter, deque
    
    grid = np.array(grid_list)
    H, W = grid.shape
    out = grid.copy()
    
    bg = Counter(int(v) for v in grid.flatten()).most_common(1)[0][0]
    
    # Find divider color
    non_bg = set(int(v) for v in grid.flatten() if v != bg)
    divider_color = None
    for color in non_bg:
        count = int(np.sum(grid == color))
        if count >= min(H, W):
            divider_color = color
            break
    if divider_color is None:
        return grid_list
    
    is_divider = (grid == divider_color)
    
    # Find sections via flood fill
    visited = np.zeros((H, W), dtype=bool)
    sections = []
    
    for sr in range(H):
        for sc in range(W):
            if visited[sr][sc] or is_divider[sr][sc]:
                continue
            region = set()
            stack = [(sr, sc)]
            while stack:
                r, c = stack.pop()
                if r < 0 or r >= H or c < 0 or c >= W or visited[r][c] or is_divider[r][c]:
                    continue
                visited[r][c] = True
                region.add((r, c))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    stack.append((r+dr, c+dc))
            if region:
                sections.append(region)
    
    for region in sections:
        # Find markers
        markers = []
        for r, c in region:
            v = int(grid[r][c])
            if v != bg and v != divider_color:
                markers.append((r, c, v))
        
        if not markers:
            continue
        
        # BFS distance from boundary (cells adjacent to divider or grid edge)
        dist = {}
        q = deque()
        
        for r, c in region:
            is_boundary = False
            if r == 0 or r == H-1 or c == 0 or c == W-1:
                is_boundary = True
            else:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if (nr, nc) not in region:
                        is_boundary = True
                        break
            if is_boundary:
                dist[(r, c)] = 0
                q.append((r, c))
        
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if (nr, nc) in region and (nr, nc) not in dist:
                    dist[(nr, nc)] = dist[(r, c)] + 1
                    q.append((nr, nc))
        
        # Find corner marker (at boundary, dist=0)
        corners = [(min(r for r,c in region), min(c for r,c in region)),
                   (min(r for r,c in region), max(c for r,c in region)),
                   (max(r for r,c in region), min(c for r,c in region)),
                   (max(r for r,c in region), max(c for r,c in region))]
        
        markers.sort(key=lambda m: dist.get((m[0], m[1]), 999))
        
        outer_color = markers[0][2]
        inner_color = markers[1][2] if len(markers) > 1 else outer_color
        
        # Fill based on BFS distance
        for r, c in region:
            d = dist.get((r, c), 0)
            if d % 2 == 0:
                out[r][c] = outer_color
            else:
                out[r][c] = inner_color
    
    return out.tolist()
