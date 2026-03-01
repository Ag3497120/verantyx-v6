
def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    # find background color (most common)
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # find rectangular 0-regions (where grid is 0 or non-bg non-zero)
    # Actually windows are where values != bg
    # Find connected rectangular regions by flood-filling non-bg areas
    visited = [[False]*cols for _ in range(rows)]
    windows = []
    
    def find_rect(sr, sc):
        # Find the bounding rectangle of connected non-bg region
        r1,r2,c1,c2 = sr,sr,sc,sc
        stack = [(sr,sc)]
        visited[sr][sc] = True
        cells = [(sr,sc)]
        while stack:
            r,c = stack.pop()
            r1 = min(r1,r); r2 = max(r2,r)
            c1 = min(c1,c); c2 = max(c2,c)
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc = r+dr,c+dc
                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] != bg:
                    visited[nr][nc] = True
                    stack.append((nr,nc))
                    cells.append((nr,nc))
        return r1,r2,c1,c2, cells
    
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != bg:
                r1,r2,c1,c2,cells = find_rect(r,c)
                windows.append((r1,r2,c1,c2,cells))
    
    if len(windows) != 2:
        return grid
    
    w1r1,w1r2,w1c1,w1c2,w1cells = windows[0]
    w2r1,w2r2,w2c1,w2c2,w2cells = windows[1]
    
    # Extract contents of each window
    def get_content(r1,r2,c1,c2):
        return [[grid[r][c] for c in range(c1,c2+1)] for r in range(r1,r2+1)]
    
    c1 = get_content(w1r1,w1r2,w1c1,w1c2)
    c2 = get_content(w2r1,w2r2,w2c1,w2c2)
    
    # Count non-bg non-zero cells in each
    def count_shape(content):
        return sum(1 for row in content for v in row if v != 0)
    
    n1, n2 = count_shape(c1), count_shape(c2)
    
    # Simpler: marker-only window has 1 non-zero cell
    if n1 <= n2:
        marker_win = c1; marker_r1,marker_c1 = w1r1,w1c1
        shape_win = c2; shape_r1,shape_c1 = w2r1,w2c1
    else:
        marker_win = c2; marker_r1,marker_c1 = w2r1,w2c1
        shape_win = c1; shape_r1,shape_c1 = w1r1,w1c1
    
    # Find marker position in marker_win
    mrow, mcol = -1, -1
    mval = 0
    for r, row in enumerate(marker_win):
        for c, v in enumerate(row):
            if v != 0:
                mrow, mcol, mval = r, c, v
    
    # Find same value in shape_win
    srow, scol = -1, -1
    for r, row in enumerate(shape_win):
        for c, v in enumerate(row):
            if v == mval:
                srow, scol = r, c
    
    if mrow < 0 or srow < 0:
        return grid
    
    # Output = marker_win sized, place shape_win with offset so marker aligns
    out_rows = len(marker_win)
    out_cols = len(marker_win[0])
    result = [[0]*out_cols for _ in range(out_rows)]
    
    # shift: shape position (srow,scol) -> output position (mrow,mcol)
    dr = mrow - srow
    dc = mcol - scol
    
    for r, row in enumerate(shape_win):
        for c, v in enumerate(row):
            if v != 0:
                nr, nc = r + dr, c + dc
                if 0 <= nr < out_rows and 0 <= nc < out_cols:
                    result[nr][nc] = v
    
    return result
