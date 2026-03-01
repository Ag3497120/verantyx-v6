
def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find all block positions (2x2 non-zero groups)
    # Identify block size: find smallest repeating non-zero patterns
    # Find all non-zero cells
    nonzero = set((r,c) for r in range(rows) for c in range(cols) if grid[r][c] != 0)
    if not nonzero:
        return grid
    
    # Find 8-blocks and non-8 blocks using connected components
    visited = set()
    blocks = []
    for r,c in sorted(nonzero):
        if (r,c) in visited:
            continue
        # BFS
        comp = []
        queue = [(r,c)]
        visited.add((r,c))
        while queue:
            cr,cc = queue.pop()
            comp.append((cr,cc))
            for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr,nc = cr+dr,cc+dc
                if (nr,nc) in nonzero and (nr,nc) not in visited:
                    visited.add((nr,nc))
                    queue.append((nr,nc))
        color = grid[comp[0][0]][comp[0][1]]
        r_min = min(r for r,c in comp)
        r_max = max(r for r,c in comp)
        c_min = min(c for r,c in comp)
        c_max = max(c for r,c in comp)
        blocks.append({'color': color, 'r0': r_min, 'r1': r_max, 'c0': c_min, 'c1': c_max})
    
    result = [row[:] for row in grid]
    
    # Find the non-8 block (should be one)
    special = [b for b in blocks if b['color'] != 8]
    eights = [b for b in blocks if b['color'] == 8]
    
    def overlaps_rows(b1, b2):
        return b1['r0'] <= b2['r1'] and b2['r0'] <= b1['r1']
    
    def overlaps_cols(b1, b2):
        return b1['c0'] <= b2['c1'] and b2['c0'] <= b1['c1']
    
    def fill_h_gap(b1, b2):
        # b1 is left of b2, same row band
        r0 = max(b1['r0'], b2['r0'])
        r1 = min(b1['r1'], b2['r1'])
        c0 = b1['c1'] + 1
        c1 = b2['c0'] - 1
        for r in range(r0, r1+1):
            for c in range(c0, c1+1):
                result[r][c] = 7
    
    def fill_v_gap(b1, b2):
        # b1 is above b2, same col band
        r0 = b1['r1'] + 1
        r1 = b2['r0'] - 1
        c0 = max(b1['c0'], b2['c0'])
        c1 = min(b1['c1'], b2['c1'])
        for r in range(r0, r1+1):
            for c in range(c0, c1+1):
                result[r][c] = 7
    
    def no_block_between_h(b1, b2, all_blocks):
        # Check if any block is between b1 and b2 horizontally (same row band)
        for b in all_blocks:
            if b is b1 or b is b2:
                continue
            if overlaps_rows(b, b1) and b['c0'] > b1['c1'] and b['c1'] < b2['c0']:
                return False
        return True
    
    def no_block_between_v(b1, b2, all_blocks):
        for b in all_blocks:
            if b is b1 or b is b2:
                continue
            if overlaps_cols(b, b1) and b['r0'] > b1['r1'] and b['r1'] < b2['r0']:
                return False
        return True
    
    # Connect the special block to the nearest 8 in each direction
    for sp in special:
        best = {}
        for b in eights:
            # Check right: b is to right of sp, overlapping rows
            if b['c0'] > sp['c1'] and overlaps_rows(sp, b):
                gap = b['c0'] - sp['c1'] - 1
                if 'R' not in best or gap < best['R'][0]:
                    if no_block_between_h(sp, b, blocks):
                        best['R'] = (gap, b)
            # Check left
            if b['c1'] < sp['c0'] and overlaps_rows(sp, b):
                gap = sp['c0'] - b['c1'] - 1
                if 'L' not in best or gap < best['L'][0]:
                    if no_block_between_h(b, sp, blocks):
                        best['L'] = (gap, b)
            # Check down
            if b['r0'] > sp['r1'] and overlaps_cols(sp, b):
                gap = b['r0'] - sp['r1'] - 1
                if 'D' not in best or gap < best['D'][0]:
                    if no_block_between_v(sp, b, blocks):
                        best['D'] = (gap, b)
            # Check up
            if b['r1'] < sp['r0'] and overlaps_cols(sp, b):
                gap = sp['r0'] - b['r1'] - 1
                if 'U' not in best or gap < best['U'][0]:
                    if no_block_between_v(b, sp, blocks):
                        best['U'] = (gap, b)
        
        # Connect to nearest in each direction
        if best:
            nearest_gap = min(v[0] for v in best.values())
            for d, (gap, b) in best.items():
                if gap == nearest_gap:
                    if d == 'R':
                        fill_h_gap(sp, b)
                    elif d == 'L':
                        fill_h_gap(b, sp)
                    elif d == 'D':
                        fill_v_gap(sp, b)
                    elif d == 'U':
                        fill_v_gap(b, sp)
    
    # Connect 8-blocks to each other (nearest in each direction, no block between)
    for i, b1 in enumerate(eights):
        for b2 in eights[i+1:]:
            # Horizontal
            if overlaps_rows(b1, b2):
                left, right = (b1, b2) if b1['c1'] < b2['c0'] else (b2, b1)
                if left['c1'] < right['c0'] and no_block_between_h(left, right, blocks):
                    # Check no special between them
                    has_special = any(overlaps_rows(sp, left) and sp['c0'] > left['c1'] and sp['c1'] < right['c0'] 
                                    for sp in special)
                    if not has_special:
                        fill_h_gap(left, right)
            # Vertical
            if overlaps_cols(b1, b2):
                top, bot = (b1, b2) if b1['r1'] < b2['r0'] else (b2, b1)
                if top['r1'] < bot['r0'] and no_block_between_v(top, bot, blocks):
                    has_special = any(overlaps_cols(sp, top) and sp['r0'] > top['r1'] and sp['r1'] < bot['r0']
                                    for sp in special)
                    if not has_special:
                        fill_v_gap(top, bot)
    
    return result
