def transform(grid):
    # Find connected segments of each non-bg color, follow the path chain
    from collections import deque, Counter
    rows, cols = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    
    # Find all non-bg cells
    cells = {(r,c): grid[r][c] for r in range(rows) for c in range(cols) if grid[r][c] != bg}
    
    # BFS to find connected components by color
    visited = set()
    segments = []  # (color, cells_set)
    
    def nbr4(r, c):
        return [(r+dr,c+dc) for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                if 0<=r+dr<rows and 0<=c+dc<cols]
    
    for pos, val in cells.items():
        if pos not in visited:
            comp = []
            q = deque([pos])
            visited.add(pos)
            while q:
                r, c = q.popleft()
                comp.append((r,c))
                for nr, nc in nbr4(r, c):
                    if (nr,nc) not in visited and grid[nr][nc] == val:
                        visited.add((nr,nc))
                        q.append((nr,nc))
            segments.append((val, set(comp)))
    
    # Build adjacency between segments
    seg_index = {}
    for i, (val, comp) in enumerate(segments):
        for pos in comp:
            seg_index[pos] = i
    
    adj = {i: set() for i in range(len(segments))}
    for i, (val, comp) in enumerate(segments):
        for r, c in comp:
            for nr, nc in nbr4(r, c):
                if (nr,nc) in seg_index and seg_index[(nr,nc)] != i:
                    adj[i].add(seg_index[(nr,nc)])
    
    # Find chain order (linear path through segments)
    # Start from endpoint (degree 1)
    ends = [i for i, neighbors in adj.items() if len(neighbors) == 1]
    if not ends:
        ends = list(adj.keys())
    
    start = ends[0]
    order = [start]
    prev = -1
    curr = start
    while True:
        nexts = [n for n in adj[curr] if n != prev]
        if not nexts:
            break
        prev = curr
        curr = nexts[0]
        order.append(curr)
    
    # Output: for each segment in order, output its color repeated by its size
    result = []
    for i in order:
        val, comp = segments[i]
        for _ in range(len(comp)):
            result.append([val])
    
    return result
