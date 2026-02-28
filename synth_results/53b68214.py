
def transform(grid):
    import numpy as np
    g = np.array(grid)
    in_rows, cols = g.shape
    # Output is 10 rows
    out_rows = 10
    result = np.zeros((out_rows, cols), dtype=int)
    result[:in_rows] = g
    
    # Find period of pattern
    def find_period(g):
        n = len(g)
        for p in range(1, n+1):
            ok = True
            for i in range(p, n):
                if not np.array_equal(g[i], g[i % p]):
                    ok = False
                    break
            if ok:
                return p
        return n
    
    period = find_period(g)
    for r in range(in_rows, out_rows):
        result[r] = g[r % period]
    return result.tolist()
