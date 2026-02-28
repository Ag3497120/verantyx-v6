def transform(grid):
    W = len(grid[0])
    result = []
    for row in grid:
        N = next((c for c in range(len(row)) if row[c] == 0), len(row))
        pat = row[:N]
        mid_len = W - N - (N - 1)
        new_row = list(pat) + [pat[0]] * mid_len + list(pat[1:])
        result.append(new_row)
    return result
