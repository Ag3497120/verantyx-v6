def transform(grid):
    from collections import Counter, deque
    R, C = len(grid), len(grid[0])
    flat = [grid[r][c] for r in range(R) for c in range(C)]
    cnt = Counter(flat)
    bg = cnt.most_common(1)[0][0]
    non_bg = [k for k in cnt if k != bg]
    # Find border color (most common non-bg) and fill/marker color
    counts_nb = [(cnt[k], k) for k in non_bg]
    counts_nb.sort(reverse=True)
    border_color = counts_nb[0][1]
    # Marker cell: smallest non-bg that's inside
    # (any non-border, non-bg cell)
    marker_cells = [(r,c) for r in range(R) for c in range(C) 
                    if grid[r][c] != bg and grid[r][c] != border_color]
    if not marker_cells:
        # Only one non-bg: flood fill from within border
        fill_color = border_color
        # Find enclosed cells via flood from exterior
        visited = [[False]*C for _ in range(R)]
        q = deque()
        for r in range(R):
            for c in range(C):
                if (r==0 or r==R-1 or c==0 or c==C-1) and grid[r][c]==bg:
                    if not visited[r][c]:
                        visited[r][c]=True
                        q.append((r,c))
        while q:
            r,c = q.popleft()
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc = r+dr,c+dc
                if 0<=nr<R and 0<=nc<C and not visited[nr][nc] and grid[nr][nc]==bg:
                    visited[nr][nc]=True
                    q.append((nr,nc))
        out = [row[:] for row in grid]
        for r in range(R):
            for c in range(C):
                if not visited[r][c] and grid[r][c]==bg:
                    out[r][c] = fill_color
        return out
    else:
        # Flood fill from marker cells through bg cells
        fill_color = grid[marker_cells[0][0]][marker_cells[0][1]]
        visited = [[False]*C for _ in range(R)]
        q = deque()
        for r,c in marker_cells:
            visited[r][c]=True
            q.append((r,c))
        # Also seed from adjacent bg cells
        for r,c in marker_cells:
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc = r+dr,c+dc
                if 0<=nr<R and 0<=nc<C and not visited[nr][nc] and grid[nr][nc]==bg:
                    visited[nr][nc]=True
                    q.append((nr,nc))
        while q:
            r,c = q.popleft()
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc = r+dr,c+dc
                if 0<=nr<R and 0<=nc<C and not visited[nr][nc] and grid[nr][nc]==bg:
                    visited[nr][nc]=True
                    q.append((nr,nc))
        out = [row[:] for row in grid]
        for r in range(R):
            for c in range(C):
                if visited[r][c] and grid[r][c]==bg:
                    out[r][c] = fill_color
        return out
