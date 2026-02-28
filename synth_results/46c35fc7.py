def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [list(row) for row in grid]
    bg = 7
    
    # Find each connected non-background patch and rotate it 180 degrees in place
    from collections import deque
    visited = [[False]*cols for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                # BFS
                q = deque([(r, c)])
                visited[r][c] = True
                cells = [(r, c)]
                while q:
                    rr, cc = q.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = rr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] != bg:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                            cells.append((nr, nc))
                
                r1 = min(rr for rr,cc in cells)
                r2 = max(rr for rr,cc in cells)
                c1 = min(cc for rr,cc in cells)
                c2 = max(cc for rr,cc in cells)
                
                # Rotate 180 degrees (flip both horizontally and vertically)
                # For each cell (rr, cc), its 180-rotated position is (r1+r2-rr, c1+c2-cc)
                new_vals = {}
                for rr, cc in cells:
                    new_r = r1 + r2 - rr
                    new_c = c1 + c2 - cc
                    new_vals[(new_r, new_c)] = grid[rr][cc]
                
                # Clear old cells first (set to bg)
                for rr, cc in cells:
                    result[rr][cc] = bg
                
                # Place new values
                for (rr, cc), v in new_vals.items():
                    result[rr][cc] = v
    
    return result
