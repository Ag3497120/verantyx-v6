def transform(grid):
    from collections import Counter, deque
    rows = len(grid)
    cols = len(grid[0])
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find connected regions per value
    visited = [[False]*cols for _ in range(rows)]
    regions = {}  # value -> list of region-cells-lists
    
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != bg:
                q = deque([(r, c)])
                visited[r][c] = True
                cells = [(r, c)]
                v = grid[r][c]
                while q:
                    cr, cc = q.popleft()
                    for dr2, dc2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr2, cc+dc2
                        if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == v:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                            cells.append((nr, nc))
                if v not in regions:
                    regions[v] = []
                regions[v].append(cells)
    
    result = [row[:] for row in grid]
    
    for v, rlist in regions.items():
        if len(rlist) < 2:
            continue
        # Sort by size - largest is block
        rlist.sort(key=lambda x: -len(x))
        block_cells = set(map(tuple, rlist[0]))
        isolated_list = rlist[1:]  # smaller regions
        
        for isolated_cells in isolated_list:
            if len(isolated_cells) != 1:
                continue  # only handle single cells for now
            ir, ic = isolated_cells[0]
            
            # Find direction toward block (nearest block cell)
            block_rows = [r for r,c in block_cells]
            block_cols = [c for r,c in block_cells]
            br_min, br_max = min(block_rows), max(block_rows)
            bc_min, bc_max = min(block_cols), max(block_cols)
            
            # Determine direction (horizontal or vertical)
            if ir < br_min:
                dr2, dc2 = 1, 0  # go down
            elif ir > br_max:
                dr2, dc2 = -1, 0  # go up
            elif ic < bc_min:
                dr2, dc2 = 0, 1  # go right
            elif ic > bc_max:
                dr2, dc2 = 0, -1  # go left
            else:
                continue
            
            # Move from isolated cell toward block until hitting it
            # Erase start
            result[ir][ic] = bg
            
            # Trace path
            r, c = ir, ic
            path = []
            while True:
                nr, nc = r + dr2, c + dc2
                if not (0 <= nr < rows and 0 <= nc < cols):
                    break
                if (nr, nc) in block_cells:
                    break  # stop before block
                path.append((nr, nc))
                r, c = nr, nc
            
            # Mark path
            for pr, pc in path:
                result[pr][pc] = v
            
            # Start arms (perpendicular and backward)
            back_dir = (-dr2, -dc2)
            perp_dirs = [(dc2, dr2), (-dc2, -dr2)]
            for ard, acd in [back_dir] + perp_dirs:
                nr2, nc2 = ir + ard, ic + acd
                if 0 <= nr2 < rows and 0 <= nc2 < cols and result[nr2][nc2] == bg:
                    result[nr2][nc2] = v
            
            # End arms (perpendicular to movement at landing)
            if path:
                lr, lc = path[-1]
                for ard, acd in perp_dirs:
                    nr2, nc2 = lr + ard, lc + acd
                    if 0 <= nr2 < rows and 0 <= nc2 < cols and result[nr2][nc2] == bg:
                        result[nr2][nc2] = v
    
    return result
