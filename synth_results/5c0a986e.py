import numpy as np

def transform(grid):
    g = np.array(grid)
    H, W = g.shape
    out = np.copy(g)
    
    # Find blocks
    blocks = {}
    for v in set(g.flat):
        if v != 0:
            cells = np.argwhere(g == v)
            r0, c0 = cells.min(0)
            r1, c1 = cells.max(0)
            blocks[int(v)] = (int(r0), int(c0), int(r1), int(c1))
    
    # Determine direction by position: block closer to top-left gets UP-LEFT trail
    color_list = sorted(blocks.keys())
    # Assign UP-LEFT (-1,-1) to the block in top-left region
    # and DOWN-RIGHT (+1,+1) to the other
    b0 = blocks[color_list[0]]
    b1 = blocks[color_list[1]]
    
    # First color: trail goes UP-LEFT from top-left corner
    def emit_trail(block, color, dr, dc):
        if dr == -1:
            start_r = block[0] + dr  # just above top
        else:
            start_r = block[2] + dr  # just below bottom
        if dc == -1:
            start_c = block[1] + dc  # just left of left edge
        else:
            start_c = block[3] + dc  # just right of right edge
        r, c = start_r, start_c
        while 0 <= r < H and 0 <= c < W:
            if g[r, c] == 0:
                out[r, c] = color
            r += dr
            c += dc
    
    emit_trail(blocks[color_list[0]], color_list[0], -1, -1)
    emit_trail(blocks[color_list[1]], color_list[1], 1, 1)
    
    return out.tolist()
