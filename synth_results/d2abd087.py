def transform(grid):
    rows = len(grid); cols = len(grid[0])
    result = [[0]*cols for _ in range(rows)]
    visited = set()
    def bfs(sr, sc):
        cells = []; q = [(sr,sc)]; visited.add((sr,sc))
        while q:
            r,c = q.pop(); cells.append((r,c))
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc = r+dr,c+dc
                if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in visited and grid[nr][nc]==5:
                    visited.add((nr,nc)); q.append((nr,nc))
        return cells
    for r in range(rows):
        for c in range(cols):
            if grid[r][c]==5 and (r,c) not in visited:
                cells = bfs(r,c)
                color = 2 if len(cells)==6 else 1
                for cr,cc in cells: result[cr][cc] = color
    return result
