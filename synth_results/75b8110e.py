def transform(grid):
    import numpy as np
    arr = np.array(grid)
    H, W = arr.shape
    h, w = H//2, W//2
    TL = arr[:h, :w]
    TR = arr[:h, w:]
    BL = arr[h:, :w]
    BR = arr[h:, w:]
    result = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            tl, tr, bl, br = TL[r,c], TR[r,c], BL[r,c], BR[r,c]
            # Priority: TR > BL > BR > TL
            if tr != 0:
                result[r,c] = tr
            elif bl != 0:
                result[r,c] = bl
            elif br != 0:
                result[r,c] = br
            elif tl != 0:
                result[r,c] = tl
    return result.tolist()