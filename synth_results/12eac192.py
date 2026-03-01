import numpy as np

def transform(grid):
    arr = np.array(grid)
    rows, cols = arr.shape
    result = arr.copy()
    
    visited = set()
    for r in range(rows):
        for c in range(cols):
            v = arr[r,c]
            if v == 0 or (r,c) in visited:
                continue
            # BFS to find connected component
            comp = []
            stack = [(r,c)]
            while stack:
                rr,cc = stack.pop()
                if (rr,cc) in visited or not(0<=rr<rows) or not(0<=cc<cols): continue
                if arr[rr,cc] != v: continue
                visited.add((rr,cc)); comp.append((rr,cc))
                for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                    stack.append((rr+dr,cc+dc))
            if len(comp) < 3:
                for rr,cc in comp:
                    result[rr,cc] = 3
    
    return result.tolist()
