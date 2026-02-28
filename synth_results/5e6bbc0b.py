
def transform(grid):
    import numpy as np
    g = np.array(grid)
    h, w = g.shape
    
    # Find the 8
    pos = list(zip(*np.where(g == 8)))
    if not pos: return g.tolist()
    r8, c8 = pos[0]
    
    # Determine what value 8 replaced (checkerboard parity)
    # Find the checkerboard: first cell (0,0) value determines parity
    # Count 0s and 1s to determine which value starts at (0,0)
    # If (r+c)%2 == 0 -> base value; else -> 1-base
    # The 8 is at (r8,c8), parity = (r8+c8)%2
    # Find the base value
    ones = int(np.sum(g == 1))
    zeros = int(np.sum(g == 0))
    # Look at a non-8 cell to determine base
    base = int(g[0, 0]) if g[0, 0] != 8 else int(g[0, 1]) if w > 1 else 0
    # Value 8 replaced:
    replaced_parity = (r8 + c8) % 2
    # base is at parity 0 (if g[0,0]=base, parity of (0,0)=0)
    base_parity = 0  # g[0,0] has parity 0
    replaced = base if replaced_parity == base_parity else 1 - base
    
    # Determine division direction: 8 is on which edge?
    dist_top = r8
    dist_bottom = h - 1 - r8
    dist_left = c8
    dist_right = w - 1 - c8
    min_dist = min(dist_top, dist_bottom, dist_left, dist_right)
    
    out = np.zeros_like(g)
    
    if min(dist_top, dist_bottom) <= min(dist_left, dist_right):
        # Horizontal division (by row)
        center = h / 2
        if r8 <= center:
            # 8 in top half -> top is 8's side (fill with 1-replaced)
            for r in range(h):
                for c in range(w):
                    if r8 <= center - 0.5:  # top half
                        if r < h/2:
                            out[r, c] = (1 - replaced) if g[r, c] != 8 else 8
                        else:
                            out[r, c] = replaced
                    else:
                        if r >= h/2:
                            out[r, c] = (1 - replaced) if g[r, c] != 8 else 8
                        else:
                            out[r, c] = replaced
            # Place 9s
            # Mirror of r8 through center: 2*center - r8
            mirror_r = 2 * center - r8
            mirror_r = max(0, min(h-1, round(mirror_r)))
            # 9s in c8's column, in the opposite half, between center and mirror
            if r8 < center:
                # 8 in top, opposite is bottom. Between center and mirror_r.
                r_start = int(center + 0.5)
                for r in range(r_start, mirror_r + 1):
                    if 0 <= r < h:
                        out[r, c8] = 9
            else:
                # 8 in bottom, opposite is top. Between center and mirror_r.
                r_end = int(center - 0.5)
                for r in range(mirror_r, r_end + 1):
                    if 0 <= r < h:
                        out[r, c8] = 9
    else:
        # Vertical division (by column)
        center = w / 2
        if c8 <= center:
            # 8 in left half -> left is 8's side
            for r in range(h):
                for c in range(w):
                    if c < w/2:
                        out[r, c] = (1 - replaced) if g[r, c] != 8 else 8
                    else:
                        out[r, c] = replaced
        else:
            for r in range(h):
                for c in range(w):
                    if c >= w/2:
                        out[r, c] = (1 - replaced) if g[r, c] != 8 else 8
                    else:
                        out[r, c] = replaced
        # Place 9s
        mirror_c = 2 * center - c8
        mirror_c = max(0, min(w-1, round(mirror_c)))
        if c8 < center:
            # 8 in left, opposite is right.
            c_start = int(center + 0.5)
            for c in range(c_start, int(mirror_c) + 1):
                if 0 <= c < w:
                    out[r8, c] = 9
        else:
            # 8 in right, opposite is left.
            c_end = int(center - 0.5)
            for c in range(int(mirror_c), c_end + 1):
                if 0 <= c < w:
                    out[r8, c] = 9
    
    # Place 8 back
    out[r8, c8] = 8
    return out.tolist()
