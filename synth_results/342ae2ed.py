from collections import defaultdict, deque, Counter

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    cnt = Counter(grid[r][c] for r in range(rows) for c in range(cols))
    bg = cnt.most_common(1)[0][0]
    visited = set()
    blobs = defaultdict(list)
    for sr in range(rows):
        for sc in range(cols):
            v = grid[sr][sc]
            if v == bg or (sr,sc) in visited: continue
            blob = []
            q = deque([(sr,sc)])
            visited.add((sr,sc))
            while q:
                r,c = q.popleft()
                blob.append((r,c))
                for dr2,dc2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr2,c+dc2
                    if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in visited and grid[nr][nc]==v:
                        visited.add((nr,nc)); q.append((nr,nc))
            blobs[v].append(blob)
    for color, blob_list in blobs.items():
        if len(blob_list) < 2: continue
        for i in range(len(blob_list)):
            for j in range(i+1, len(blob_list)):
                b1, b2 = blob_list[i], blob_list[j]
                c1r = sum(r for r,c in b1)/len(b1)
                c1c = sum(c for r,c in b1)/len(b1)
                c2r = sum(r for r,c in b2)/len(b2)
                c2c = sum(c for r,c in b2)/len(b2)
                dr = 1 if c2r > c1r else -1
                dc = 1 if c2c > c1c else -1
                if dr==1: r1 = max(r for r,c in b1)
                else: r1 = min(r for r,c in b1)
                if dc==1: c1 = max(c for r,c in b1)
                else: c1 = min(c for r,c in b1)
                if dr==1: r2 = min(r for r,c in b2)
                else: r2 = max(r for r,c in b2)
                if dc==1: c2 = min(c for r,c in b2)
                else: c2 = max(c for r,c in b2)
                cr,cc = r1+dr, c1+dc
                while (cr,cc) != (r2,c2):
                    if 0<=cr<rows and 0<=cc<cols and grid[cr][cc]==bg:
                        result[cr][cc] = color
                    cr+=dr; cc+=dc
    return result
