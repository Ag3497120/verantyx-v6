def transform(grid):
    import numpy as np
    g = np.array(grid)
    R, C = g.shape
    bg = 0
    wall = 5  # The grid color
    
    # Find rows and cols that are entirely 'wall'
    h_dividers = [r for r in range(R) if np.all(g[r] == wall)]
    v_dividers = [c for c in range(C) if np.all(g[:, c] == wall)]
    
    # Find cell row-bands and col-bands
    def get_bands(dividers, size):
        bands = []
        prev = -1
        for d in dividers:
            if d > prev + 1:
                bands.append((prev + 1, d - 1))
            prev = d
        if prev < size - 1:
            bands.append((prev + 1, size - 1))
        return bands
    
    row_bands = get_bands(h_dividers, R)
    col_bands = get_bands(v_dividers, C)
    
    N_rows = len(row_bands)
    N_cols = len(col_bands)
    
    out = g.copy()
    
    def fill_cell(cell_r, cell_c, color):
        r1, r2 = row_bands[cell_r]
        c1, c2 = col_bands[cell_c]
        out[r1:r2+1, c1:c2+1] = color
    
    fill_cell(0, 0, 1)
    fill_cell(N_rows // 2, N_cols // 2, 2)
    fill_cell(N_rows - 1, N_cols - 1, 3)
    
    return out.tolist()
