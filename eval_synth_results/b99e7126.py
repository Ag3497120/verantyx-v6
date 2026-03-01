def transform(grid):
    R, C = len(grid), len(grid[0])
    period = 4
    num_tr = R // period
    num_tc = C // period
    tile_counts = {}
    tiles = {}
    ts = period - 1
    for tr in range(num_tr):
        for tc in range(num_tc):
            content = tuple(grid[period*tr+dr][period*tc+dc] for dr in range(1, period) for dc in range(1, period))
            tile_counts[content] = tile_counts.get(content, 0) + 1
            tiles[(tr, tc)] = content
    base_content = max(tile_counts, key=tile_counts.get)
    anomalous = {}
    for (tr, tc), content in tiles.items():
        if content != base_content:
            anomalous[(tr, tc)] = content
    if not anomalous:
        return grid
    anom_content = list(anomalous.values())[0]
    base_3x3 = [list(base_content[i*ts:(i+1)*ts]) for i in range(ts)]
    anom_3x3 = [list(anom_content[i*ts:(i+1)*ts]) for i in range(ts)]
    center_val = base_3x3[ts//2][ts//2]
    preserve = set()
    for dr in range(ts):
        for dc in range(ts):
            if anom_3x3[dr][dc] == center_val:
                preserve.add((dr, dc))
    anom_positions = set(anomalous.keys())
    best_center = None
    for cr in range(num_tr):
        for cc in range(num_tc):
            valid = True
            for (tr, tc) in anom_positions:
                pr, pc = tr - (cr - 1), tc - (cc - 1)
                if pr < 0 or pr >= ts or pc < 0 or pc >= ts:
                    valid = False
                    break
                if (pr, pc) in preserve:
                    valid = False
                    break
            if valid:
                best_center = (cr, cc)
                break
        if best_center:
            break
    if not best_center:
        return grid
    cr, cc = best_center
    out = [row[:] for row in grid]
    for dr in range(ts):
        for dc in range(ts):
            tr, tc = cr - 1 + dr, cc - 1 + dc
            if tr < 0 or tr >= num_tr or tc < 0 or tc >= num_tc:
                continue
            if (dr, dc) in preserve:
                for r in range(ts):
                    for c in range(ts):
                        out[period*tr+1+r][period*tc+1+c] = base_3x3[r][c]
            else:
                for r in range(ts):
                    for c in range(ts):
                        out[period*tr+1+r][period*tc+1+c] = anom_3x3[r][c]
    return out
