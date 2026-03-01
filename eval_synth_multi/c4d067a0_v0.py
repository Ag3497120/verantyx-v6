import numpy as np
from collections import Counter

def transform(grid):
    g = np.array(grid)
    H, W = g.shape
    bg = Counter(g.flatten().tolist()).most_common(1)[0][0]
    out = g.copy()
    
    # Find connected components
    visited = np.zeros((H, W), dtype=bool)
    markers = []
    blocks = []
    
    for r in range(H):
        for c in range(W):
            if visited[r, c] or g[r, c] == bg:
                continue
            stack = [(r, c)]
            cells = []
            while stack:
                cr, cc = stack.pop()
                if cr < 0 or cc < 0 or cr >= H or cc >= W: continue
                if visited[cr, cc] or g[cr, cc] == bg: continue
                visited[cr, cc] = True
                cells.append((cr, cc))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    stack.append((cr+dr, cc+dc))
            if len(cells) == 1:
                markers.append((cells[0][0], cells[0][1], int(g[cells[0][0], cells[0][1]])))
            elif len(cells) >= 4:
                rows_b = [x[0] for x in cells]
                cols_b = [x[1] for x in cells]
                r0, r1, c0, c1 = min(rows_b), max(rows_b), min(cols_b), max(cols_b)
                bh, bw = r1 - r0 + 1, c1 - c0 + 1
                if bh * bw == len(cells):
                    blocks.append({'r0': r0, 'c0': c0, 'h': bh, 'w': bw, 
                                   'color': int(g[cells[0][0], cells[0][1]])})
    
    if not markers or not blocks:
        return out.tolist()
    
    # Build marker grid
    marker_map = {}
    for r, c, v in markers:
        marker_map[(r, c)] = v
    
    m_rows = sorted(set(r for r, c, v in markers))
    m_cols = sorted(set(c for r, c, v in markers))
    
    # Block dimensions
    block_h = blocks[0]['h']
    block_w = blocks[0]['w']
    
    # Block column starts from existing blocks
    existing_col_starts = sorted(set(b['c0'] for b in blocks))
    
    # Column spacing
    if len(existing_col_starts) >= 2:
        col_spacing = existing_col_starts[1] - existing_col_starts[0]
    else:
        col_spacing = block_w + 1
    
    row_spacing = col_spacing
    
    # Existing block row
    existing_row = blocks[0]['r0']
    existing_colors = []
    for bc in existing_col_starts:
        for b in blocks:
            if b['r0'] == existing_row and b['c0'] == bc:
                existing_colors.append(b['color'])
                break
    
    # Match existing blocks to marker row and columns
    # Try each marker row to find color match
    best_match = None
    for idx, mr in enumerate(m_rows):
        for col_offset in range(len(m_cols) - len(existing_col_starts) + 1):
            row_colors = []
            for j in range(len(existing_col_starts)):
                mc = m_cols[col_offset + j]
                if (mr, mc) in marker_map:
                    row_colors.append(marker_map[(mr, mc)])
                else:
                    row_colors.append(None)
            if row_colors == existing_colors:
                best_match = (idx, col_offset)
                break
        if best_match:
            break
    
    if best_match is None:
        # Try reverse matching or partial
        best_match = (0, 0)
    
    matched_row_idx, matched_col_offset = best_match
    
    # Calculate block start positions
    block_start_row = existing_row - matched_row_idx * row_spacing
    block_start_col = existing_col_starts[0] - matched_col_offset * col_spacing
    
    # Fill in all blocks
    for idx, mr in enumerate(m_rows):
        block_r = block_start_row + idx * row_spacing
        for jdx, mc in enumerate(m_cols):
            if (mr, mc) not in marker_map:
                continue
            color = marker_map[(mr, mc)]
            block_c = block_start_col + jdx * col_spacing
            
            for dr in range(block_h):
                for dc in range(block_w):
                    r, c = block_r + dr, block_c + dc
                    if 0 <= r < H and 0 <= c < W:
                        out[r, c] = color
    
    return out.tolist()
