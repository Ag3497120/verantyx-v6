
def transform(grid):
    import numpy as np
    from collections import Counter, deque
    g = np.array(grid)
    bg = Counter(g.flatten().tolist()).most_common(1)[0][0]
    # Find non-bg cells
    rows, cols = np.where(g != bg)
    if len(rows) == 0:
        return g.tolist()
    cells = set(zip(rows.tolist(), cols.tolist()))
    # Build path by BFS from an endpoint (cell with <=1 neighbor in path)
    def neighbors(r, c):
        return [(r+dr, c+dc) for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)] if (r+dr,c+dc) in cells]
    # Find endpoint (has exactly 1 or 0 neighbor)
    start = None
    for r, c in cells:
        if len(neighbors(r, c)) <= 1:
            start = (r, c)
            break
    if start is None:
        start = list(cells)[0]
    # Traverse path
    path = []
    visited = set()
    current = start
    while current is not None:
        path.append(current)
        visited.add(current)
        nxt = None
        for nb in neighbors(*current):
            if nb not in visited:
                nxt = nb
                break
        current = nxt
    # Extract values and reverse
    values = [int(g[r, c]) for r, c in path]
    rev_values = values[::-1]
    out = g.copy()
    for i, (r, c) in enumerate(path):
        out[r, c] = rev_values[i]
    return out.tolist()
