import numpy as np

def transform(grid):
    g = np.array(grid)
    H, W = g.shape
    twos = np.argwhere(g == 2)
    r0, c0 = twos.min(0)
    r1, c1 = twos.max(0)
    frame_h = int(r1 - r0 + 1)
    frame_w = int(c1 - c0 + 1)
    int_h = frame_h - 2
    int_w = frame_w - 2
    
    # Extract frame into output
    out = np.zeros((frame_h, frame_w), int)
    out[0, :] = 2; out[-1, :] = 2
    out[:, 0] = 2; out[:, -1] = 2
    
    # Find small shape
    colors = set(int(v) for v in g.flat if v not in (0, 2))
    for color in colors:
        cells = np.argwhere(g == color)
        sr, sc = cells.min(0)
        er, ec = cells.max(0)
        shape_h = int(er - sr + 1)
        shape_w = int(ec - sc + 1)
        sub = g[sr:er+1, sc:ec+1]
        scale_h = int_h // shape_h
        scale_w = int_w // shape_w
        # Scale the shape
        scaled = np.repeat(np.repeat(sub, scale_h, axis=0), scale_w, axis=1)
        out[1:frame_h-1, 1:frame_w-1] = scaled
    
    return out.tolist()
