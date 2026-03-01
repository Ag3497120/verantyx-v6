def transform(grid):
    from collections import Counter
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]
    bg = grid[0][0]
    border_val = 2

    content_rows = []
    for r in range(rows):
        row = grid[r]
        if border_val in row:
            first_2 = row.index(border_val)
            last_2 = len(row) - 1 - row[::-1].index(border_val)
            if first_2 != last_2:
                interior = row[first_2+1:last_2]
                if not all(v == border_val for v in interior):
                    content_rows.append((r, first_2, last_2))

    for r, c1, c2 in content_rows:
        interior = grid[r][c1+1:c2]
        n = len(interior)

        already_ok = False
        for period in range(1, n // 2 + 1):
            pattern = [interior[i % period] for i in range(period)]
            tiled = [pattern[i % period] for i in range(n)]
            if tiled == interior:
                already_ok = True
                break
        if already_ok:
            continue

        best_period = None
        best_pattern = None
        best_mismatches = n + 1

        for period in range(1, n // 2 + 1):
            pattern = []
            for p in range(period):
                vals = [interior[i] for i in range(p, n, period)]
                cnt = Counter(vals)
                pattern.append(cnt.most_common(1)[0][0])
            tiled = [pattern[i % period] for i in range(n)]
            mismatches = sum(1 for i in range(n) if tiled[i] != interior[i])
            if 0 < mismatches < best_mismatches:
                best_mismatches = mismatches
                best_period = period
                best_pattern = pattern

        if best_pattern:
            new_interior = [best_pattern[i % best_period] for i in range(n)]
            out[r] = grid[r][:c1+1] + new_interior + grid[r][c2:]

    return out
