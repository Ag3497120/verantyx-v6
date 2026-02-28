from collections import deque

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [list(row) for row in grid]
    bg = 7
    
    # Find connected components of non-background cells
    # For each component, find outlier cells (those touching the border of bg specially)
    # The rule: each shape has one cell with value 8 (marker) - remove that cell
    # by setting it and the shape it's adjacent to back to bg, then reposition the shape
    
    visited = [[False]*cols for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in [bg, 8] and not visited[r][c]:
                v = grid[r][c]
                q = deque([(r, c)])
                visited[r][c] = True
                cells = [(r, c)]
                while q:
                    rr, cc = q.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = rr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] not in [bg]:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                            cells.append((nr, nc))
                
                # Find the 8 in this component
                eights = [(rr,cc) for rr,cc in cells if grid[rr][cc] == 8]
                non_eights = [(rr,cc) for rr,cc in cells if grid[rr][cc] != 8]
                
                if not eights:
                    continue
                
                # Find bounding box of non-8 cells
                if not non_eights:
                    continue
                
                er, ec = eights[0]  # position of 8
                
                # Find center of the non-8 part
                r1 = min(rr for rr,cc in non_eights)
                r2 = max(rr for rr,cc in non_eights)
                c1 = min(cc for rr,cc in non_eights)
                c2 = max(cc for rr,cc in non_eights)
                
                center_r = (r1 + r2) / 2
                center_c = (c1 + c2) / 2
                
                # Clear entire component
                for rr, cc in cells:
                    result[rr][cc] = bg
                
                # Determine shift: the 8 indicates where to move the shape
                # Move opposite to the 8's direction from center
                # If 8 is at top-left of shape, shape moves top-left
                # Actually: the 8 is an "extra" cell that indicates direction
                # The shape moves so the 8 connects to the shape's edge
                
                # Simple: place non-8 cells in their original position (keep them)
                # just remove the 8
                for rr, cc in non_eights:
                    result[rr][cc] = v
    
    return result
