from collections import deque, defaultdict

def transform(grid):
    R, C = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    WIRE = 6; SEG = 2
    
    def add_wire_down(r_start, c):
        if not (0 <= c < C): return
        for r in range(r_start, R):
            if grid[r][c] == SEG: break
            result[r][c] = WIRE
    
    def get_connected_segments(val):
        visited = [[False]*C for _ in range(R)]
        segments = []
        for r in range(R):
            for c in range(C):
                if grid[r][c] == val and not visited[r][c]:
                    seg = []
                    q = deque([(r,c)])
                    visited[r][c] = True
                    while q:
                        cr,cc = q.popleft()
                        seg.append((cr,cc))
                        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr,nc = cr+dr,cc+dc
                            if 0<=nr<R and 0<=nc<C and not visited[nr][nc] and grid[nr][nc]==val:
                                visited[nr][nc] = True
                                q.append((nr,nc))
                    segments.append(seg)
        return segments
    
    for r in range(R):
        for c in range(C):
            if grid[r][c] == WIRE:
                add_wire_down(r, c)
    
    segs = get_connected_segments(SEG)
    seg_by_toprow = defaultdict(list)
    for seg in segs:
        if len(seg) <= 1: continue
        min_r = min(r for r,c in seg)
        if min_r < 1: continue
        seg_by_toprow[min_r - 1].append(seg)
    
    for r_top in range(R):
        if r_top not in seg_by_toprow: continue
        for seg in seg_by_toprow[r_top]:
            min_c = min(c for r,c in seg)
            max_c = max(c for r,c in seg)
            c_left = min_c - 1; c_right = max_c + 1
            if not any(result[r_top][c] == WIRE for c in range(min_c, max_c+1)): continue
            for c in range(max(0,c_left), min(C,c_right+1)):
                if grid[r_top][c] != SEG: result[r_top][c] = WIRE
            add_wire_down(r_top, c_left)
            add_wire_down(r_top, c_right)
    
    return result
