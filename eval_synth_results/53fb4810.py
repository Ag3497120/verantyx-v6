def transform(grid):
    R = len(grid)
    C = len(grid[0])
    bg = 8
    out = [row[:] for row in grid]
    
    # Find all cross/plus shapes made of 1s
    # Find connected components of 1s
    visited = [[False]*C for _ in range(R)]
    crosses = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 1 and not visited[r][c]:
                # BFS
                comp = []
                stack = [(r,c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr,cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and grid[nr][nc] == 1:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                crosses.append(comp)
    
    # For each cross, find adjacent non-bg, non-1 cells (markers/tails)
    for cross in crosses:
        cross_set = set(cross)
        # Find the bounding box
        min_r = min(r for r,c in cross)
        max_r = max(r for r,c in cross)
        min_c = min(c for r,c in cross)
        max_c = max(c for r,c in cross)
        
        # Find non-bg, non-1 neighbors
        neighbors = []
        for cr, cc in cross:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr+dr, cc+dc
                if 0 <= nr < R and 0 <= nc < C and (nr,nc) not in cross_set:
                    v = grid[nr][nc]
                    if v != bg and v != 1:
                        neighbors.append((nr, nc, v))
        
        if not neighbors:
            continue
        
        # Check if this is a tail (long pattern) or marker (short)
        # Find all non-bg non-1 cells connected to this cross (not just adjacent)
        tail_cells = set()
        for nr, nc, v in neighbors:
            if (nr, nc) not in tail_cells:
                # BFS for non-bg non-1 cells
                stack2 = [(nr, nc)]
                tail_cells.add((nr, nc))
                while stack2:
                    tr, tc = stack2.pop()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ntr, ntc = tr+dr, tc+dc
                        if 0 <= ntr < R and 0 <= ntc < C and (ntr, ntc) not in tail_cells and (ntr, ntc) not in cross_set:
                            if grid[ntr][ntc] != bg and grid[ntr][ntc] != 1:
                                tail_cells.add((ntr, ntc))
                                stack2.append((ntr, ntc))
        
        if len(tail_cells) <= 4:  # This is a marker, need to extend
            # Determine direction from cross to marker
            marker_cells = sorted(tail_cells)
            marker_min_r = min(r for r,c in marker_cells)
            marker_max_r = max(r for r,c in marker_cells)
            marker_min_c = min(c for r,c in marker_cells)
            marker_max_c = max(c for r,c in marker_cells)
            
            # Direction: which side of the cross is the marker on?
            if marker_max_r < min_r:  # marker above cross
                direction = 'up'
            elif marker_min_r > max_r:  # marker below
                direction = 'down'
            elif marker_max_c < min_c:  # marker left
                direction = 'left'
            elif marker_min_c > max_c:  # marker right
                direction = 'right'
            else:
                # Marker overlaps with cross bounds - check relative position
                avg_mr = sum(r for r,c in marker_cells) / len(marker_cells)
                avg_mc = sum(c for r,c in marker_cells) / len(marker_cells)
                avg_cr = sum(r for r,c in cross) / len(cross)
                avg_cc = sum(c for r,c in cross) / len(cross)
                if avg_mr < avg_cr:
                    direction = 'up'
                elif avg_mr > avg_cr:
                    direction = 'down'
                elif avg_mc < avg_cc:
                    direction = 'left'
                else:
                    direction = 'right'
            
            # Get the pattern columns/rows of the marker
            if direction in ('up', 'down'):
                # Marker spans some columns, pattern is per-column
                cols_used = sorted(set(c for r,c in marker_cells))
                # For each column, get the pattern of colors going away from cross
                if direction == 'up':
                    # Sort rows descending (closest to cross first)
                    pattern_rows = sorted(set(r for r,c in marker_cells), reverse=True)
                    # Get pattern per column
                    col_patterns = {}
                    for col in cols_used:
                        pat = []
                        for row in pattern_rows:
                            if (row, col) in tail_cells:
                                pat.append(grid[row][col])
                        col_patterns[col] = pat
                    
                    # Extend upward from marker to edge
                    start_row = marker_min_r - 1
                    for row in range(start_row, -1, -1):
                        dist = start_row - row
                        for col in cols_used:
                            pat = col_patterns[col]
                            if pat:
                                out[row][col] = pat[dist % len(pat)]
                else:  # down
                    pattern_rows = sorted(set(r for r,c in marker_cells))
                    col_patterns = {}
                    for col in cols_used:
                        pat = []
                        for row in pattern_rows:
                            if (row, col) in tail_cells:
                                pat.append(grid[row][col])
                        col_patterns[col] = pat
                    
                    start_row = marker_max_r + 1
                    for row in range(start_row, R):
                        dist = row - start_row
                        for col in cols_used:
                            pat = col_patterns[col]
                            if pat:
                                out[row][col] = pat[dist % len(pat)]
            
            else:  # left or right
                rows_used = sorted(set(r for r,c in marker_cells))
                if direction == 'right':
                    pattern_cols = sorted(set(c for r,c in marker_cells))
                    row_patterns = {}
                    for row in rows_used:
                        pat = []
                        for col in pattern_cols:
                            if (row, col) in tail_cells:
                                pat.append(grid[row][col])
                        row_patterns[row] = pat
                    
                    start_col = marker_max_c + 1
                    for col in range(start_col, C):
                        dist = col - start_col
                        for row in rows_used:
                            pat = row_patterns[row]
                            if pat:
                                out[row][col] = pat[dist % len(pat)]
                else:  # left
                    pattern_cols = sorted(set(c for r,c in marker_cells), reverse=True)
                    row_patterns = {}
                    for row in rows_used:
                        pat = []
                        for col in pattern_cols:
                            if (row, col) in tail_cells:
                                pat.append(grid[row][col])
                        row_patterns[row] = pat
                    
                    start_col = marker_min_c - 1
                    for col in range(start_col, -1, -1):
                        dist = start_col - col
                        for row in rows_used:
                            pat = row_patterns[row]
                            if pat:
                                out[row][col] = pat[dist % len(pat)]
    
    return out
