def transform(grid):
    return _solve(grid)

def solve_6e02f1e3(grid):
    flat = [v for row in grid for v in row]
    distinct = len(set(flat))
    if distinct == 1:
        return [[5,5,5],[0,0,0],[0,0,0]]
    elif distinct == 2:
        return [[5,0,0],[0,5,0],[0,0,5]]
    else:
        return [[0,0,5],[0,5,0],[5,0,0]]


_solve = solve_6e02f1e3
