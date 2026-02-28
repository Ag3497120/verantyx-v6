from collections import Counter
def transform(grid):
    H = len(grid)
    W = len(grid[0])
    cnt = Counter(v for row in grid for v in row)
    bg = cnt.most_common(1)[0][0]
    first_row = {}
    for r, row in enumerate(grid):
        for v in row:
            if v != bg and v not in first_row:
                first_row[v] = r
    sorted_colors = sorted(first_row, key=first_row.get)
    out_colors = [bg] * (5 - len(sorted_colors)) + sorted_colors
    return [[c, c, c] for c in out_colors]
