
def transform(grid):
    import numpy as np
    from scipy import ndimage
    from collections import Counter
    g = np.array(grid)
    h, w = g.shape
    
    # Background = most common value
    cnt = Counter(g.flatten().tolist())
    bg = cnt.most_common(1)[0][0]
    
    colors = sorted(set(g.flatten().tolist()) - {bg})
    
    # Count connected components per color
    color_counts = {}
    for color in colors:
        mask = (g == color).astype(int)
        _, n = ndimage.label(mask)
        color_counts[color] = n
    
    if not color_counts:
        return [[]]
    
    max_count = max(color_counts.values())
    out = []
    for color in colors:
        row = [color] * color_counts[color] + [0] * (max_count - color_counts[color])
        out.append(row)
    
    return out
