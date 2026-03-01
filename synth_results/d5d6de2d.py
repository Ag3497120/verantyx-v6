def transform(grid):
    rows = len(grid); cols = len(grid[0])
    result = [[0]*cols for _ in range(rows)]
    cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c]==2]
    if not cells: return result
    visited = set()
    def bfs(sr, sc):
        q = [(sr,sc)]; visited.add((sr,sc)); group = []
        while q:
            r,c = q.pop(); group.append((r,c))
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc = r+dr,c+dc
                if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in visited and grid[nr][nc]==2:
                    visited.add((nr,nc)); q.append((nr,nc))
        return group
    groups = []
    for r,c in cells:
        if (r,c) not in visited: groups.append(bfs(r,c))
    for frame in groups:
        minr=min(r for r,c in frame); maxr=max(r for r,c in frame)
        minc=min(c for r,c in frame); maxc=max(c for r,c in frame)
        if maxr > minr+1 and maxc > minc+1:
            for r in range(minr+1, maxr):
                for c in range(minc+1, maxc): result[r][c] = 3
    return result
