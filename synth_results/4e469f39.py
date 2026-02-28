def transform(grid):
    rows, cols = len(grid), len(grid[0])
    result = [list(row) for row in grid]
    
    # Find connected components of 5-cells
    visited = [[False]*cols for _ in range(rows)]
    components = []
    
    for sr in range(rows):
        for sc in range(cols):
            if grid[sr][sc] == 5 and not visited[sr][sc]:
                comp = []
                stack = [(sr,sc)]
                while stack:
                    r,c = stack.pop()
                    if visited[r][c]: continue
                    visited[r][c] = True
                    comp.append((r,c))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc = r+dr,c+dc
                        if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]==5 and not visited[nr][nc]:
                            stack.append((nr,nc))
                components.append(comp)
    
    for comp in components:
        min_r = min(r for r,c in comp)
        max_r = max(r for r,c in comp)
        min_c = min(c for r,c in comp)
        max_c = max(c for r,c in comp)
        
        top_row, bottom_row, left_col, right_col = min_r, max_r, min_c, max_c
        
        # Find gap in top row
        gap_col = None
        for c in range(left_col+1, right_col):
            if grid[top_row][c] == 0:
                gap_col = c
                break
        
        if gap_col is None:
            continue
        
        # Fill interior with 2s
        for r in range(top_row+1, bottom_row):
            for c in range(left_col+1, right_col):
                if result[r][c] == 0:
                    result[r][c] = 2
        result[top_row][gap_col] = 2
        
        # Draw ray at row above bracket
        ray_row = top_row - 1
        if ray_row >= 0:
            dist_left = gap_col - left_col
            dist_right = right_col - gap_col
            if dist_left <= dist_right:
                for c in range(gap_col, cols):
                    result[ray_row][c] = 2
            else:
                for c in range(0, gap_col+1):
                    result[ray_row][c] = 2
    
    return result
