def transform(grid):
    from collections import Counter
    inp = [list(r) for r in grid]
    R, C = len(inp), len(inp[0])
    bg = Counter(v for row in inp for v in row).most_common(1)[0][0]
    center_r = max(range(R), key=lambda r: len(set(v for v in inp[r] if v != bg)))
    center_row = inp[center_r]
    # Horizontal bar color (most common in center row)
    hbar_color = Counter(center_row).most_common(1)[0][0]
    
    # Arm groups: contiguous columns with same non-hbar_color value in center row
    arm_groups = []
    visited = set()
    for c in range(C):
        if c in visited:
            continue
        val = center_row[c]
        if val == hbar_color:
            visited.add(c)
            continue
        group_cols = [c]
        visited.add(c)
        nc = c + 1
        while nc < C and center_row[nc] == val:
            group_cols.append(nc)
            visited.add(nc)
            nc += 1
        extent = 0
        for gc in group_cols:
            val_rows = [r for r in range(R) if inp[r][gc] != bg and r != center_r]
            if val_rows:
                e = max(abs(r - center_r) for r in val_rows)
                extent = max(extent, e)
        arm_groups.append((group_cols, val, extent))
    arm_groups.sort(key=lambda x: x[0][0])
    extents = [g[2] for g in arm_groups]
    sorted_extents = sorted(extents)
    out = [row[:] for row in inp]
    for i, (group_cols, val, old_extent) in enumerate(arm_groups):
        new_extent = sorted_extents[i]
        for c in group_cols:
            for r in range(R):
                if r != center_r:
                    out[r][c] = bg
            for d in range(1, new_extent + 1):
                if center_r - d >= 0:
                    out[center_r - d][c] = val
                if center_r + d < R:
                    out[center_r + d][c] = val
    return out
