def transform(grid):
    rows, cols = len(grid), len(grid[0])
    g = [list(row) for row in grid]
    layers = []
    seen = set()
    for d in range(min(rows,cols)//2 + 1):
        if 2*d >= rows or 2*d >= cols: break
        color = g[d][d]
        if color not in seen:
            layers.append(color)
            seen.add(color)
    if len(layers) < 2: return grid
    color_map = {}
    n = len(layers)
    for i in range(n):
        color_map[layers[i]] = layers[(i-1) % n]
    return [[color_map.get(v, v) for v in row] for row in grid]
