import numpy as np

def transform(grid):
    arr = np.array(grid)
    rows, cols = arr.shape
    result = arr.copy()
    bg = 7; mark = 2
    
    visited = set()
    for start_r in range(rows):
        for start_c in range(cols):
            v = arr[start_r, start_c]
            if v == bg or v == mark or (start_r, start_c) in visited: continue
            comp = []; stack = [(start_r,start_c)]
            while stack:
                r,c = stack.pop()
                if (r,c) in visited or not(0<=r<rows) or not(0<=c<cols): continue
                if arr[r,c] != v: continue
                visited.add((r,c)); comp.append((r,c))
                for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]: stack.append((r+dr,c+dc))
            if not comp: continue
            rs = [r for r,c in comp]; cs = [c for r,c in comp]
            rmin,rmax = min(rs),max(rs); cmin,cmax = min(cs),max(cs)
            h = rmax-rmin+1; w = cmax-cmin+1
            if h < 2 or w < 2: continue  # need at least 2x2
            expected = set((r,c) for r in range(rmin,rmax+1) for c in range(cmin,cmax+1) if r==rmin or r==rmax or c==cmin or c==cmax)
            if set(comp) != expected: continue
            # Valid if has interior (h>=3, w>=3) or is 2x2 square
            if not ((h >= 3 and w >= 3) or (h == 2 and w == 2)): continue
            for (cr,cc) in [(rmin,cmin),(rmin,cmax),(rmax,cmin),(rmax,cmax)]:
                dr = -1 if cr==rmin else 1
                dc = -1 if cc==cmin else 1
                if 0<=cr+dr<rows: result[cr+dr,cc] = mark
                if 0<=cc+dc<cols: result[cr,cc+dc] = mark
    
    return result.tolist()
