from collections import Counter

def transform(grid):
    rows = len(grid); cols = len(grid[0])
    flat = [v for r in grid for v in r]
    tc = Counter(flat)
    order = sorted(tc.keys(), key=lambda x: (-tc[x], x))
    seq = []
    for v in order:
        seq.extend([v] * tc[v])
    result = [[0]*cols for _ in range(rows)]
    idx = 0
    for c in range(cols):
        for r in range(rows-1, -1, -1):
            if idx < len(seq):
                result[r][c] = seq[idx]
                idx += 1
    return result
