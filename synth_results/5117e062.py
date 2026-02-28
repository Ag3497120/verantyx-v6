
def transform(grid):
    import numpy as np
    g = np.array(grid)
    # Find the 8
    pos = list(zip(*np.where(g == 8)))
    if not pos:
        return grid
    r, c = pos[0]
    # Get the 3x3 around it
    r0, c0 = r-1, c-1
    patch = g[r0:r0+3, c0:c0+3].copy()
    # Determine color (most common non-zero, non-8)
    vals = patch.flatten()
    color = 0
    for v in vals:
        if v != 0 and v != 8:
            color = v
            break
    # Replace 8 with color
    patch[patch == 8] = color
    return patch.tolist()
