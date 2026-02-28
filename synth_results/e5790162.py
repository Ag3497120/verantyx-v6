def transform(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    start = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3:
                start = (r, c)
    
    if not start:
        return grid
    
    # Global flag: does 8 appear in input?
    has_eight = any(grid[r][c] == 8 for r in range(rows) for c in range(cols))
    
    result = [row[:] for row in grid]
    r, c = start
    dr, dc = 0, 1  # start RIGHT
    
    used_markers = set()  # markers used as obstacles (stopped before)
    
    for _ in range(rows * cols * 4):
        # Find next obstacle in direction, skipping used markers
        nr, nc = r + dr, c + dc
        path = []
        obstacle_pos = None
        while 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc] != 0 and (nr, nc) not in used_markers:
                obstacle_pos = (nr, nc)
                break
            path.append((nr, nc))
            nr += dr
            nc += dc
        
        for pr, pc in path:
            result[pr][pc] = 3
        
        if path:
            r, c = path[-1]
        
        if obstacle_pos is None:
            break
        
        used_markers.add(obstacle_pos)
        
        perp1 = (-dc, dr)   # CCW
        perp2 = (dc, -dr)   # CW
        
        def find_marker(from_r, from_c, d_r, d_c):
            tr, tc = from_r + d_r, from_c + d_c
            while 0 <= tr < rows and 0 <= tc < cols:
                if grid[tr][tc] != 0 and (tr, tc) not in used_markers:
                    return (tr, tc)
                tr += d_r
                tc += d_c
            return None
        
        m1 = find_marker(r, c, perp1[0], perp1[1])
        m2 = find_marker(r, c, perp2[0], perp2[1])
        
        if m1 and not m2:
            chosen = perp1
        elif m2 and not m1:
            chosen = perp2
        elif m1 and m2:
            d1 = abs(m1[0]-r) + abs(m1[1]-c)
            d2 = abs(m2[0]-r) + abs(m2[1]-c)
            chosen = perp1 if d1 <= d2 else perp2
        else:
            def dist_wall(from_r, from_c, d_r, d_c):
                steps = 0
                tr, tc = from_r + d_r, from_c + d_c
                while 0 <= tr < rows and 0 <= tc < cols:
                    steps += 1
                    tr += d_r
                    tc += d_c
                return steps
            d1 = dist_wall(r, c, perp1[0], perp1[1])
            d2 = dist_wall(r, c, perp2[0], perp2[1])
            if has_eight:
                chosen = perp1 if d1 <= d2 else perp2  # shorter
            else:
                chosen = perp1 if d1 >= d2 else perp2  # farther
        
        dr, dc = chosen
    
    return result
