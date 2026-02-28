from collections import deque

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [list(row) for row in grid]
    
    # Find each frame of 1s with a single special cell inside it
    # The special cell's color fills the interior of the frame
    visited = [[False]*cols for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and not visited[r][c]:
                q = deque([(r, c)])
                visited[r][c] = True
                cells = [(r, c)]
                while q:
                    rr, cc = q.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = rr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == 1:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                            cells.append((nr, nc))
                
                r1 = min(rr for rr,cc in cells)
                r2 = max(rr for rr,cc in cells)
                c1 = min(cc for rr,cc in cells)
                c2 = max(cc for rr,cc in cells)
                
                # Find special cell inside the frame
                special_color = None
                for rr in range(r1, r2+1):
                    for cc in range(c1, c2+1):
                        if grid[rr][cc] != 0 and grid[rr][cc] != 1:
                            special_color = grid[rr][cc]
                            break
                    if special_color:
                        break
                
                if special_color is None:
                    continue
                
                # Add a row of special_color above the frame
                if r1 - 1 >= 0:
                    for cc in range(c1, c2+1):
                        result[r1-1][cc] = special_color
                
                # Fill interior with special_color
                for rr in range(r1, r2+1):
                    for cc in range(c1, c2+1):
                        if grid[rr][cc] == 0 or grid[rr][cc] == special_color:
                            result[rr][cc] = special_color
                        elif grid[rr][cc] == 1:
                            pass  # keep 1s
    
    return result
