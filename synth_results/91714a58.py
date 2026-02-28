def transform(grid):
    import numpy as np
    g = np.array(grid)
    R, C = g.shape
    
    def max_rect_of_color(v):
        # Find largest rectangle filled with v using histogram approach
        binary = (g == v).astype(int)
        best = (0, 0, 0, 0, 0)  # (area, r1, c1, r2, c2)
        # heights[c] = consecutive v's ending at current row
        heights = np.zeros(C, dtype=int)
        for r in range(R):
            heights = np.where(binary[r] == 1, heights + 1, 0)
            # Largest rectangle in histogram
            stack = []
            for c in range(C + 1):
                h = heights[c] if c < C else 0
                start = c
                while stack and stack[-1][1] > h:
                    sc, sh = stack.pop()
                    area = sh * (c - sc)
                    if area > best[0]:
                        r1 = r - sh + 1
                        best = (area, r1, sc, r, c - 1)
                    start = sc
                stack.append((start, h))
        return best
    
    best_area = 0
    best_block = None
    best_v = 0
    for v in np.unique(g):
        if v == 0:
            continue
        area, r1, c1, r2, c2 = max_rect_of_color(v)
        if area > best_area:
            best_area = area
            best_block = (r1, c1, r2, c2)
            best_v = v
    
    out = np.zeros_like(g)
    if best_block:
        r1, c1, r2, c2 = best_block
        out[r1:r2+1, c1:c2+1] = best_v
    return out.tolist()
