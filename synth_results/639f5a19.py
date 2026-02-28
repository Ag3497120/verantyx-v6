import numpy as np

def transform(grid):
    g = np.array(grid)
    out = np.zeros_like(g)
    H, W = g.shape
    
    # Find all 8-rectangles (solid blocks)
    visited = np.zeros((H, W), bool)
    for sr in range(H):
        for sc in range(W):
            if g[sr, sc] == 8 and not visited[sr, sc]:
                # BFS to find full rectangle
                r0, c0, r1, c1 = sr, sc, sr, sc
                for r in range(sr, H):
                    if g[r, sc] == 8:
                        r1 = r
                    else:
                        break
                for c in range(sc, W):
                    if g[sr, c] == 8:
                        c1 = c
                    else:
                        break
                
                visited[r0:r1+1, c0:c1+1] = True
                
                # Border thickness
                t = 2
                # Midpoints (inclusive ranges)
                mid_r = (r0 + r1) / 2
                mid_c = (c0 + c1) / 2
                
                # Fill colors: TL=6, TR=1, BL=2, BR=3, center=4
                for r in range(r0, r1+1):
                    for c in range(c0, c1+1):
                        in_top = r < r0 + t
                        in_bot = r > r1 - t
                        in_left = c < c0 + t
                        in_right = c > c1 - t
                        is_left_half = c <= mid_c
                        is_top_half = r <= mid_r
                        
                        if in_top:
                            out[r, c] = 6 if is_left_half else 1
                        elif in_bot:
                            out[r, c] = 2 if is_left_half else 3
                        elif in_left:
                            out[r, c] = 6 if is_top_half else 2
                        elif in_right:
                            out[r, c] = 1 if is_top_half else 3
                        else:
                            out[r, c] = 4
    
    return out.tolist()
