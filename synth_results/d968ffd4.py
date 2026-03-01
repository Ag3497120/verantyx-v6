from collections import defaultdict, Counter

def transform(grid):
    rows = len(grid); cols = len(grid[0])
    result = [row[:] for row in grid]
    flat = [v for r in grid for v in r]
    bg = Counter(flat).most_common(1)[0][0]
    colors = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg: colors[grid[r][c]].append((r,c))
    colorlist = list(colors.keys())
    if len(colorlist) < 2: return result
    c1, c2 = colorlist[0], colorlist[1]
    cells1 = colors[c1]; cells2 = colors[c2]
    minr1=min(r for r,c in cells1); maxr1=max(r for r,c in cells1)
    minc1=min(c for r,c in cells1); maxc1=max(c for r,c in cells1)
    minr2=min(r for r,c in cells2); maxr2=max(r for r,c in cells2)
    minc2=min(c for r,c in cells2); maxc2=max(c for r,c in cells2)
    if maxc1 < minc2:
        N = (minc2-maxc1-1)//2
        for r in range(rows):
            for c in range(maxc1+1, maxc1+N+1): result[r][c] = c1
            for c in range(minc2-N, minc2): result[r][c] = c2
    elif maxc2 < minc1:
        N = (minc1-maxc2-1)//2
        for r in range(rows):
            for c in range(maxc2+1, maxc2+N+1): result[r][c] = c2
            for c in range(minc1-N, minc1): result[r][c] = c1
    elif maxr1 < minr2:
        N = (minr2-maxr1-1)//2
        for c in range(cols):
            for r in range(maxr1+1, maxr1+N+1): result[r][c] = c1
            for r in range(minr2-N, minr2): result[r][c] = c2
    elif maxr2 < minr1:
        N = (minr1-maxr2-1)//2
        for c in range(cols):
            for r in range(maxr2+1, maxr2+N+1): result[r][c] = c2
            for r in range(minr1-N, minr1): result[r][c] = c1
    return result
