import numpy as np

def transform(grid):
    g = np.array(grid)
    out = g.copy()
    h, w = g.shape
    
    def get(r, c):
        if 0 <= r < h and 0 <= c < w:
            return int(g[r, c])
        return -1
    
    for r in range(h):
        for c in range(w):
            if g[r, c] == 1:
                continue
            # Check all 4 arms at distance 2 with domino pointing away
            up_near = get(r-2, c) == 1
            up_far = get(r-3, c) == 1
            down_near = get(r+2, c) == 1
            down_far = get(r+3, c) == 1
            left_near = get(r, c-2) == 1
            left_far = get(r, c-3) == 1
            right_near = get(r, c+2) == 1
            right_far = get(r, c+3) == 1
            
            if (up_near and up_far and down_near and down_far and
                left_near and left_far and right_near and right_far):
                out[r, c] = 4
    
    return out.tolist()
