def transform(grid):
    # For each row: find main color (not 0, not 7)
    # Output: color fills the non-main positions, main positions become 0
    g = [list(r) for r in grid]
    rows, cols = len(g), len(g[0])
    for r in range(rows):
        row = g[r]
        from collections import Counter
        cnt = Counter(v for v in row if v != 0 and v != 7)
        if not cnt:
            continue
        main_color = cnt.most_common(1)[0][0]
        new_row = [main_color if row[c] != main_color else 0 for c in range(cols)]
        g[r] = new_row
    return g
