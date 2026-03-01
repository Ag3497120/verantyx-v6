def transform(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = 7
    sep = 6
    
    # Find separator rows and cols
    sep_rows = [r for r in range(h) if all(g[r][c] == sep for c in range(w))]
    sep_cols = [c for c in range(w) if all(g[r][c] == sep for r in range(h))]
    
    # Extract section ranges
    def get_ranges(seps, total):
        ranges = []
        prev = -1
        for s in sorted(seps):
            if s > prev + 1:
                ranges.append((prev + 1, s))
            prev = s
        if prev + 1 < total:
            ranges.append((prev + 1, total))
        return ranges
    
    row_ranges = get_ranges(sep_rows, h)
    col_ranges = get_ranges(sep_cols, w)
    
    # Extract sections
    sections = []
    for rr in row_ranges:
        for cr in col_ranges:
            section = g[rr[0]:rr[1], cr[0]:cr[1]].copy()
            area = int(np.sum(section != bg))
            sections.append({'data': section, 'area': area, 'pos': (rr, cr)})
    
    n_row_sections = len(row_ranges)
    n_col_sections = len(col_ranges)
    section_h = row_ranges[0][1] - row_ranges[0][0]
    section_w = col_ranges[0][1] - col_ranges[0][0]
    
    is_2d = n_row_sections > 1 and n_col_sections > 1
    is_vertical = n_row_sections > 1 and n_col_sections == 1
    is_horizontal = n_row_sections == 1 and n_col_sections > 1
    
    if is_2d:
        # Sort by area ascending, output as vertical
        sections.sort(key=lambda s: s['area'])
        n = len(sections)
        out_h = n * section_h + (n - 1)
        out_w = section_w
        out = np.full((out_h, out_w), bg, dtype=int)
        for i, s in enumerate(sections):
            r_start = i * (section_h + 1)
            out[r_start:r_start + section_h, :] = s['data']
            if i < n - 1:
                out[r_start + section_h, :] = sep
        return out.tolist()
    
    elif is_vertical:
        # Rotate 90° CW: vertical → horizontal, reverse order
        ordered = list(reversed(sections))
        n = len(ordered)
        out_h = section_h
        out_w = n * section_w + (n - 1)
        out = np.full((out_h, out_w), bg, dtype=int)
        for i, s in enumerate(ordered):
            c_start = i * (section_w + 1)
            out[:, c_start:c_start + section_w] = s['data']
            if i < n - 1:
                out[:, c_start + section_w] = sep
        return out.tolist()
    
    elif is_horizontal:
        # Rotate 90° CW: horizontal → vertical, preserve order
        ordered = sections
        n = len(ordered)
        out_h = n * section_h + (n - 1)
        out_w = section_w
        out = np.full((out_h, out_w), bg, dtype=int)
        for i, s in enumerate(ordered):
            r_start = i * (section_h + 1)
            out[r_start:r_start + section_h, :] = s['data']
            if i < n - 1:
                out[r_start + section_h, :] = sep
        return out.tolist()
    
    return grid
