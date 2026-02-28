
def transform(grid):
    import numpy as np
    from collections import Counter
    g = np.array(grid)
    bg = Counter(g.flatten().tolist()).most_common(1)[0][0]
    colors = [c for c in set(g.flatten().tolist()) if c != bg]
    sizes = [(int((g == c).sum()), c) for c in colors]
    sizes.sort(reverse=True)  # largest first
    n = len(sizes)
    out_size = 2 * n - 1
    out = np.zeros((out_size, out_size), dtype=int)
    for r in range(out_size):
        for c in range(out_size):
            dist = min(r, c, out_size - 1 - r, out_size - 1 - c)
            # dist=0 -> outermost (sizes[0]=largest), dist=n-1 -> center (sizes[n-1]=smallest)
            out[r, c] = sizes[dist][1]
    return out.tolist()
