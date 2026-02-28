
def transform(grid):
    import numpy as np
    g = np.array(grid)
    h, w = g.shape
    special = {0, 8}
    # Find legend: 2x2 block in corner with no 0 or 8
    legend = None
    legend_r, legend_c = None, None
    for r0 in [0, h-2]:
        for c0 in [0, w-2]:
            block = g[r0:r0+2, c0:c0+2]
            if not any(v in special for v in block.flatten()):
                legend = block
                legend_r, legend_c = r0, c0
                break
        if legend is not None:
            break
    if legend is None:
        return g.tolist()
    tl, tr = int(legend[0, 0]), int(legend[0, 1])
    bl, br = int(legend[1, 0]), int(legend[1, 1])
    # Find main area bounds using 0-cells (main bg)
    zero_rows, zero_cols = np.where(g == 0)
    mr0, mr1 = int(zero_rows.min()), int(zero_rows.max())
    mc0, mc1 = int(zero_cols.min()), int(zero_cols.max())
    mr_mid = (mr0 + mr1) / 2
    mc_mid = (mc0 + mc1) / 2
    # Color sparse cells
    out = g.copy()
    non_leg_mask = np.ones((h, w), dtype=bool)
    non_leg_mask[legend_r:legend_r+2, legend_c:legend_c+2] = False
    sparse_mask = non_leg_mask & ~np.isin(g, list(special))
    sparse_rows, sparse_cols = np.where(sparse_mask)
    for r, c in zip(sparse_rows, sparse_cols):
        if r <= mr_mid and c <= mc_mid:
            out[r, c] = tl
        elif r <= mr_mid and c > mc_mid:
            out[r, c] = tr
        elif r > mr_mid and c <= mc_mid:
            out[r, c] = bl
        else:
            out[r, c] = br
    return out.tolist()
