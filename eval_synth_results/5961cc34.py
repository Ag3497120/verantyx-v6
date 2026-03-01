import json
from collections import deque

with open('/private/tmp/arc-agi-2/data/evaluation/5961cc34.json') as f:
    data = json.load(f)

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    bg = 8
    
    pos_4 = None
    pos_2s = []
    shape_cells = set()
    marker_cells = set()
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 4:
                pos_4 = (r, c)
            elif grid[r][c] == 2:
                pos_2s.append((r, c))
            elif grid[r][c] == 1:
                shape_cells.add((r, c))
            elif grid[r][c] == 3:
                marker_cells.add((r, c))
    
    # Direction from 2s toward 4 (continuing past 4)
    avg_2r = sum(r for r, c in pos_2s) / len(pos_2s)
    avg_2c = sum(c for r, c in pos_2s) / len(pos_2s)
    dr = pos_4[0] - avg_2r
    dc = pos_4[1] - avg_2c
    if abs(dr) >= abs(dc):
        direction = (-1, 0) if dr < 0 else (1, 0)
    else:
        direction = (0, -1) if dc < 0 else (0, 1)
    
    # Find shape connected components
    visited = set()
    shapes = []
    for cell in shape_cells:
        if cell in visited:
            continue
        queue = deque([cell])
        component = {cell}
        while queue:
            r, c = queue.popleft()
            for dr2, dc2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr2, c+dc2
                if (nr, nc) in shape_cells and (nr, nc) not in component:
                    component.add((nr, nc))
                    queue.append((nr, nc))
        
        markers = set()
        for r, c in component:
            for dr2, dc2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr2, c+dc2
                if (nr, nc) in marker_cells:
                    markers.add((nr, nc))
        
        # Determine exit direction from markers
        exit_dir = None
        if markers:
            min_r = min(r for r, c in component)
            max_r = max(r for r, c in component)
            min_c = min(c for r, c in component)
            max_c = max(c for r, c in component)
            center_r = (min_r + max_r) / 2
            center_c = (min_c + max_c) / 2
            ml = list(markers)
            avg_mr = sum(r for r, c in ml) / len(ml)
            avg_mc = sum(c for r, c in ml) / len(ml)
            mdr = avg_mr - center_r
            mdc = avg_mc - center_c
            if abs(mdr) >= abs(mdc):
                exit_dir = (-1, 0) if mdr < 0 else (1, 0)
            else:
                exit_dir = (0, -1) if mdc < 0 else (0, 1)
        
        shapes.append({
            'cells': component,
            'markers': markers,
            'exit_dir': exit_dir
        })
        visited |= component
    
    # Build output
    output = [[bg] * cols for _ in range(rows)]
    
    # Place seed
    for r, c in pos_2s:
        output[r][c] = 2
    output[pos_4[0]][pos_4[1]] = 2
    
    all_fill = set()
    connected_ids = set()
    
    def extend_pipe(positions, d):
        dr, dc = d
        pipe = []
        cur = list(positions)
        while True:
            nxt = [(r + dr, c + dc) for r, c in cur]
            if any(r < 0 or r >= rows or c < 0 or c >= cols for r, c in nxt):
                break
            hit = None
            for shape in shapes:
                for pos in nxt:
                    if pos in shape['cells'] or pos in shape['markers']:
                        hit = shape
                        break
                if hit:
                    break
            if hit:
                return pipe, hit
            pipe.extend(nxt)
            cur = nxt
        # No shape hit, fill to edge
        pipe.extend(nxt_valid := [(r,c) for r,c in nxt if 0<=r<rows and 0<=c<cols])
        # Actually we need to continue to edge
        # The nxt that went out of bounds means we stop
        # But some might still be in bounds
        # Let me redo: continue until ALL out of bounds
        # Actually need to handle properly
        return pipe, None
    
    # Hmm, let me redo extend_pipe more carefully
    def extend_pipe2(positions, d):
        dr, dc = d
        pipe = []
        cur = list(positions)
        while True:
            nxt = [(r + dr, c + dc) for r, c in cur]
            # Filter in-bounds
            valid = [(r, c) for r, c in nxt if 0 <= r < rows and 0 <= c < cols]
            if not valid:
                break
            # Check if any valid position hits a shape
            hit = None
            for shape in shapes:
                if id(shape) in connected_ids:
                    continue
                for pos in valid:
                    if pos in shape['cells'] or pos in shape['markers']:
                        hit = shape
                        break
                if hit:
                    break
            if hit:
                return pipe, hit
            pipe.extend(valid)
            cur = valid
        return pipe, None
    
    # BFS through shapes from seed
    pipe_queue = deque()
    pipe_queue.append(([pos_4], direction))
    
    while pipe_queue:
        positions, d = pipe_queue.popleft()
        pipe, hit = extend_pipe2(positions, d)
        for r, c in pipe:
            all_fill.add((r, c))
        
        if hit and id(hit) not in connected_ids:
            connected_ids.add(id(hit))
            for r, c in hit['cells']:
                all_fill.add((r, c))
            for r, c in hit['markers']:
                all_fill.add((r, c))
            if hit['exit_dir'] and hit['markers']:
                pipe_queue.append((list(hit['markers']), hit['exit_dir']))
    
    for r, c in all_fill:
        output[r][c] = 2
    
    return output

# Test
for i, ex in enumerate(data['train']):
    result = transform(ex['input'])
    expected = ex['output']
    if result == expected:
        print(f"Train {i}: PASS")
    else:
        print(f"Train {i}: FAIL")
        for r in range(len(expected)):
            if r < len(result) and result[r] != expected[r]:
                print(f"  Row {r}: got {result[r]}")
                print(f"       exp {expected[r]}")
