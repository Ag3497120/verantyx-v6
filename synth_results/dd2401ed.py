
def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    all_five_cols = set(c for r in range(rows) for c in range(cols) if grid[r][c] == 5)
    if len(all_five_cols) == 1:
        fc = list(all_five_cols)[0]
        ones = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
        twos = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 2]
        if not ones or not twos: return grid
        avg_one_col = sum(c for r,c in ones) / len(ones)
        N = len(ones)
        if avg_one_col < fc:
            right_twos = sorted([(r,c) for r,c in twos if c > fc], key=lambda x: x[1])
            if not right_twos: return grid
            leftmost_2_col = right_twos[0][1]
            if N == 1:
                new_fc = leftmost_2_col - 1
                to_convert = set()
            else:
                new_fc = leftmost_2_col + N
                to_convert = set((r,c) for r,c in right_twos if c < new_fc)
        else:
            left_twos = sorted([(r,c) for r,c in twos if c < fc], key=lambda x: -x[1])
            if not left_twos: return grid
            rightmost_2_col = left_twos[0][1]
            if N == 1:
                new_fc = rightmost_2_col + 1
                to_convert = set()
            else:
                new_fc = rightmost_2_col - N
                to_convert = set((r,c) for r,c in left_twos if c > new_fc)
        out = [[0]*cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                v = grid[r][c]
                if v == 5: out[r][new_fc] = 5
                elif v == 2: out[r][c] = 1 if (r,c) in to_convert else 2
                elif v != 0: out[r][c] = v
        return out
    return grid
