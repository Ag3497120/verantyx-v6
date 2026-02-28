from collections import deque

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [list(row) for row in grid]
    bg = grid[0][0]  # background color
    
    visited = [[False]*cols for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and grid[r][c] != 6 and not visited[r][c]:
                frame_color = grid[r][c]
                q = deque([(r, c)])
                visited[r][c] = True
                frame_cells = set()
                frame_cells.add((r, c))
                while q:
                    rr, cc = q.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = rr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == frame_color:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                            frame_cells.add((nr, nc))
                
                if len(frame_cells) < 4:
                    continue
                
                fr1 = min(r for r,c in frame_cells)
                fr2 = max(r for r,c in frame_cells)
                fc1 = min(c for r,c in frame_cells)
                fc2 = max(c for r,c in frame_cells)
                
                fc_mid = (fc1 + fc2) // 2
                fr_mid = (fr1 + fr2) // 2
                
                # Draw vertical line at fc_mid
                for rr in range(rows):
                    if (rr, fc_mid) not in frame_cells:
                        result[rr][fc_mid] = 6
                
                # Draw horizontal line at fr_mid
                for cc in range(cols):
                    if (fr_mid, cc) not in frame_cells:
                        result[fr_mid][cc] = 6
    
    return result
