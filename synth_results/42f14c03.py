
def transform(grid):
    from collections import Counter
    rows = len(grid)
    cols = len(grid[0])
    flat = [grid[r][c] for r in range(rows) for c in range(cols)]
    bg = Counter(flat).most_common(1)[0][0]
    
    visited = [[False]*cols for _ in range(rows)]
    def bfs(sr, sc):
        color = grid[sr][sc]
        cells = []
        stack = [(sr,sc)]
        visited[sr][sc] = True
        while stack:
            r,c = stack.pop()
            cells.append((r,c))
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc = r+dr,c+dc
                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc]==color:
                    visited[nr][nc] = True
                    stack.append((nr,nc))
        return color, cells
    
    components = []  # (color, cells)
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != bg:
                color, cells = bfs(r, c)
                components.append((color, cells))
    
    def get_bbox(cells):
        rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
        return min(rs),max(rs),min(cs),max(cs)
    
    # Try each component as the frame (look for one with bg holes inside bbox)
    for fidx, (frame_color, frame_cells) in enumerate(components):
        r1,r2,c1,c2 = get_bbox(frame_cells)
        h,w = r2-r1+1, c2-c1+1
        if h < 3 or w < 3:
            continue
        content = [[grid[r][c] for c in range(c1,c2+1)] for r in range(r1,r2+1)]
        
        # Find rectangular bg regions inside
        vis2 = [[False]*w for _ in range(h)]
        bg_regions = []
        for sr in range(h):
            for sc in range(w):
                if content[sr][sc] == bg and not vis2[sr][sc]:
                    cells2 = []
                    stack = [(sr,sc)]
                    vis2[sr][sc] = True
                    while stack:
                        r,c = stack.pop()
                        cells2.append((r,c))
                        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr,nc = r+dr,c+dc
                            if 0<=nr<h and 0<=nc<w and not vis2[nr][nc] and content[nr][nc]==bg:
                                vis2[nr][nc] = True
                                stack.append((nr,nc))
                    brs = [p[0] for p in cells2]; bcs = [p[1] for p in cells2]
                    bg_regions.append((min(brs),max(brs),min(bcs),max(bcs),len(cells2)))
        
        if not bg_regions:
            continue
        
        # For each bg region, find a matching component (same h,w,count)
        fill_map = {}
        all_matched = True
        for br1,br2,bc1,bc2,bcnt in bg_regions:
            bh,bw = br2-br1+1, bc2-bc1+1
            matched = False
            for jdx, (sc, scells) in enumerate(components):
                if jdx == fidx:
                    continue
                sr1,sr2,sc1,sc2 = get_bbox(scells)
                sh,sw = sr2-sr1+1, sc2-sc1+1
                if sh == bh and sw == bw and len(scells) == bcnt:
                    fill_map[(br1,br2,bc1,bc2)] = sc
                    matched = True
                    break
            if not matched:
                all_matched = False
                break
        
        if all_matched:
            result = [row[:] for row in content]
            for (br1,br2,bc1,bc2), fc in fill_map.items():
                for r in range(br1,br2+1):
                    for c in range(bc1,bc2+1):
                        result[r][c] = fc
            return result
    
    return grid
