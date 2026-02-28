#!/usr/bin/env python3
"""Per-task solutions for batch6"""
import json, os, subprocess, numpy as np
from pathlib import Path
from collections import Counter

DATA_DIR = Path("/tmp/arc-agi-2/data/training")
RESULTS_DIR = Path(os.path.expanduser("~/verantyx_v6/synth_results"))
VERIFY_SCRIPT = Path(os.path.expanduser("~/verantyx_v6/verify_transform.py"))

def verify(tid, code):
    p = RESULTS_DIR / f"{tid}.py"
    p.write_text(code)
    try:
        r = subprocess.run(['python3', str(VERIFY_SCRIPT), str(DATA_DIR/f"{tid}.json"), str(p)],
                          capture_output=True, text=True, timeout=30)
        out = r.stdout + r.stderr
        return 'correct' in out.lower(), out
    except:
        return False, "error"

SOLUTIONS = {}

# ===== 794b24be: count 1s, place N 2s top-left in specific order =====
SOLUTIONS['794b24be'] = '''
def transform(grid):
    count = sum(v for row in grid for v in row if v == 1)
    # Order: row 0 left to right, then row 1 center first
    order = [(0,0),(0,1),(0,2),(1,1),(1,0),(1,2),(2,0),(2,1),(2,2)]
    result = [[0]*3 for _ in range(3)]
    for i in range(min(count, len(order))):
        r, c = order[i]
        result[r][c] = 2
    return result
'''

# ===== 77fdfe62: corners + interior 8s pattern =====
SOLUTIONS['77fdfe62'] = '''
def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    # Find border rows/cols of 1s
    # Row 0 and last: corners
    # The grid has a 1-border, corners have non-1 values, interior has 8s and 0s
    # Corners at (0,0), (0,-1), (-1,0), (-1,-1) might have colors
    # Actually: find rows that are all-1s (border)
    border_rows = [i for i in range(rows) if all(g[i,j]==1 for j in range(cols))]
    border_cols = [j for j in range(cols) if all(g[i,j]==1 for i in range(rows))]
    
    if not border_rows or not border_cols:
        # Simple: 1-pixel border
        r1, r2 = 0, rows-1
        c1, c2 = 0, cols-1
    else:
        r1, r2 = border_rows[0], border_rows[-1]
        c1, c2 = border_cols[0], border_cols[-1]
    
    # Extract corners (non-1 values at row border between col borders)
    # TL corner: at (r1, c1) area
    # Find corners: values that are not 0, not 1, not 8
    tl = tr = bl = br = 0
    for r in range(rows):
        for c in range(cols):
            v = g[r,c]
            if v != 0 and v != 1 and v != 8:
                # Determine which corner
                if r <= rows//2 and c <= cols//2:
                    tl = v
                elif r <= rows//2:
                    tr = v
                elif c <= cols//2:
                    bl = v
                else:
                    br = v
    
    # Extract interior (between borders)
    interior = g[r1+1:r2, c1+1:c2] if border_rows and border_cols else g[1:-1, 1:-1]
    ih, iw = interior.shape
    
    result = interior.copy()
    for r in range(ih):
        for c in range(iw):
            if interior[r,c] == 8:
                # Determine quadrant
                if r < ih//2 and c < iw//2:
                    result[r,c] = tl
                elif r < ih//2:
                    result[r,c] = tr
                elif c < iw//2:
                    result[r,c] = bl
                else:
                    result[r,c] = br
    return result.tolist()
'''

# ===== 75b8110e: 4-quadrant overlay, priority TR>BL>BR>TL =====
SOLUTIONS['75b8110e'] = '''
def transform(grid):
    import numpy as np
    arr = np.array(grid)
    H, W = arr.shape
    h, w = H//2, W//2
    TL = arr[:h, :w]
    TR = arr[:h, w:]
    BL = arr[h:, :w]
    BR = arr[h:, w:]
    result = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            tl, tr, bl, br = TL[r,c], TR[r,c], BL[r,c], BR[r,c]
            # Priority: TR > BL > BR > TL
            if tr != 0:
                result[r,c] = tr
            elif bl != 0:
                result[r,c] = bl
            elif br != 0:
                result[r,c] = br
            elif tl != 0:
                result[r,c] = tl
    return result.tolist()
'''

# ===== 6f473927: hstack with swap, direction based on edge =====
SOLUTIONS['6f473927'] = '''
def transform(grid):
    import numpy as np
    arr = np.array(grid)
    # fliplr then swap 2<->0 (2→0, 0→8... actually 0→8)
    flipped = np.fliplr(arr)
    swapped = np.where(flipped==2, 0, np.where(flipped==0, 8, flipped))
    # Determine which side to put input: left edge all 0? put input left, swap right
    left_col = arr[:,0]
    right_col = arr[:,-1]
    if all(left_col == 0):
        return np.hstack([arr, swapped]).tolist()
    else:
        return np.hstack([swapped, arr]).tolist()
'''

# ===== 7c008303: 8-separator + key + pattern, replace pattern with key colors =====
SOLUTIONS['7c008303'] = '''
def transform(grid):
    import numpy as np
    arr = np.array(grid)
    rows, cols = arr.shape
    
    # Find 8-separator row and column
    sep_rows = [i for i in range(rows) if all(arr[i,:]==8)]
    sep_cols = [j for j in range(cols) if all(arr[:,j]==8)]
    
    if not sep_rows or not sep_cols:
        return grid
    
    sr, sc = sep_rows[0], sep_cols[0]
    
    # 4 quadrants
    quads = {
        'TL': arr[:sr, :sc],
        'TR': arr[:sr, sc+1:],
        'BL': arr[sr+1:, :sc],
        'BR': arr[sr+1:, sc+1:],
    }
    
    # Find pattern quad (has 3s) and key quad (has non-0, non-8, non-3 values)
    pattern_quad = None
    key_quad = None
    pattern_name = None
    key_name = None
    
    for name, q in quads.items():
        vals = set(q.flatten()) - {0, 8}
        if 3 in vals:
            pattern_quad = q
            pattern_name = name
        elif vals:
            key_quad = q
            key_name = name
    
    if pattern_quad is None or key_quad is None:
        return grid
    
    ph, pw = pattern_quad.shape
    kh, kw = key_quad.shape
    
    # Block size
    bh = ph // kh
    bw = pw // kw
    
    result = np.zeros_like(pattern_quad)
    for r in range(ph):
        for c in range(pw):
            if pattern_quad[r, c] == 3:
                kr = r // bh
                kc = c // bw
                if kr < kh and kc < kw:
                    result[r, c] = key_quad[kr, kc]
    
    return result.tolist()
'''

# ===== 72ca375d: find most-filled object, return its bounding box =====
SOLUTIONS['72ca375d'] = '''
def transform(grid):
    import numpy as np
    from collections import Counter
    arr = np.array(grid)
    
    # Find all colored objects (connected by color, ignoring 0)
    flat = arr.flatten()
    counts = Counter(v for v in flat if v != 0)
    
    best_obj = None
    best_fill = -1
    best_box = None
    
    for color, cnt in counts.items():
        mask = (arr == color)
        rows_with = np.where(mask.any(axis=1))[0]
        cols_with = np.where(mask.any(axis=0))[0]
        if len(rows_with) == 0:
            continue
        r0, r1 = rows_with[0], rows_with[-1]
        c0, c1 = cols_with[0], cols_with[-1]
        bbox_area = (r1-r0+1) * (c1-c0+1)
        fill = cnt / bbox_area
        if fill > best_fill:
            best_fill = fill
            best_obj = color
            best_box = (r0, r1, c0, c1)
    
    if best_box is None:
        return grid
    
    r0, r1, c0, c1 = best_box
    return arr[r0:r1+1, c0:c1+1].tolist()
'''

# ===== 6e02f1e3: 3x3 symmetry analysis =====
SOLUTIONS['6e02f1e3'] = '''
def transform(grid):
    # Count 1s and map to 5s based on symmetry
    # If all same color → top row 5s
    # If left-right symmetric (all rows palindromes) → main diagonal
    # Else → anti-diagonal
    flat = [v for row in grid for v in row]
    unique = set(flat)
    
    if len(unique) == 1:
        return [[5,5,5],[0,0,0],[0,0,0]]
    
    # Check left-right symmetry (each row is a palindrome)
    lr_sym = all(row == row[::-1] for row in grid)
    
    if lr_sym:
        # Main diagonal
        return [[5,0,0],[0,5,0],[0,0,5]]
    else:
        # Anti-diagonal
        return [[0,0,5],[0,5,0],[5,0,0]]
'''

# ===== 7e4d4f7c: output 3 rows =====
SOLUTIONS['7e4d4f7c'] = '''
def transform(grid):
    # Output is always 3 rows
    # Row 0 = input row 0
    # Row 1 = input row 1
    # Row 2 = derived: where row-0 is non-bg → 6; bg stays (if col-0 != 6)
    #                   where row-0 is bg → 6; non-bg stays (if col-0 == 6)
    from collections import Counter
    row0 = grid[0]
    row1 = grid[1]
    
    # Find background of row 0
    cnt = Counter(row0)
    bg0 = cnt.most_common(1)[0][0]
    
    # Find col-0 non-bg values
    col0_vals = [grid[r][0] for r in range(1, len(grid))]
    col0_nonbg = [v for v in col0_vals if v != 0]
    col0_marker = col0_nonbg[0] if col0_nonbg else 0
    
    row2 = []
    for c in range(len(row0)):
        v = row0[c]
        if col0_marker == 6:
            # bg→6, non-bg stays
            if v == bg0:
                row2.append(6)
            else:
                row2.append(v)
        else:
            # non-bg→6, bg stays
            if v != bg0:
                row2.append(6)
            else:
                row2.append(v)
    
    return [row0, row1, row2]
'''

# ===== 73c3b0d8: 4 falls toward divider, bounces =====
# Actually needs more analysis - let me skip for now and add a placeholder
# After analyzing: each 4 at (r,c) above divider generates a diamond
# centered at (divider-1, c), expanding upward by 1 each step 

# ===== Additional tasks =====

# 72322fa7: 8 alone, surrounded by pattern → fill pattern at 8 location
SOLUTIONS['72322fa7'] = '''
def transform(grid):
    import numpy as np
    arr = np.array(grid)
    rows, cols = arr.shape
    
    # Find the template pattern (cluster of non-8, non-0 cells together with 8 in center)
    # and the lone 8s to fill with that pattern
    
    # Identify all non-zero values
    nonzero_vals = set(arr.flatten()) - {0}
    
    # Find 8 positions and other non-zero positions
    eights = list(zip(*np.where(arr == 8)))
    
    # For each non-8 non-zero color, find its shape/pattern
    templates = {}
    for val in nonzero_vals - {8}:
        positions = list(zip(*np.where(arr == val)))
        if not positions:
            continue
        # Find nearby 8
        for pr, pc in positions:
            for er, ec in eights:
                dr, dc = er - pr, ec - pc
                # Check if this 8 and these positions form a template
                # Template: 8 at center with val at offsets
                offsets = [(r-er, c-ec) for r,c in positions]
                templates[val] = (er, ec, offsets)
                break
            break
    
    result = arr.copy()
    
    # Find the template: a group containing both 8 and another color
    # Look for 8s that are adjacent to non-8 values
    template_8s = set()
    lone_8s = []
    for er, ec in eights:
        has_neighbor = False
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if dr==0 and dc==0:
                    continue
                nr, nc = er+dr, ec+dc
                if 0<=nr<rows and 0<=nc<cols and arr[nr,nc] not in [0,8]:
                    has_neighbor = True
        if has_neighbor:
            template_8s.add((er,ec))
        else:
            lone_8s.append((er,ec))
    
    # Get the template pattern from a template_8
    template_offsets = {}  # color -> list of (dr, dc) offsets from 8
    for tr, tc in template_8s:
        for val in nonzero_vals - {8}:
            positions = list(zip(*np.where(arr == val)))
            for pr, pc in positions:
                # Check if this val position is near this template 8
                max_dist = max(rows, cols)
                off = []
                for vr, vc in positions:
                    off.append((vr-tr, vc-tc))
                template_offsets[val] = off
    
    # Apply template to each lone 8
    for lr, lc in lone_8s:
        for val, offsets in template_offsets.items():
            for dr, dc in offsets:
                nr, nc = lr+dr, lc+dc
                if 0<=nr<rows and 0<=nc<cols:
                    result[nr,nc] = val
        # Remove the lone 8
        result[lr,lc] = 0
    
    # Remove the template 8s too if needed
    for tr, tc in template_8s:
        pass  # Keep them? Or remove?
    
    return result.tolist()
'''

# Let me override 72322fa7 with a better solution based on analysis
SOLUTIONS['72322fa7'] = '''
def transform(grid):
    import numpy as np
    arr = np.array(grid)
    rows, cols = arr.shape
    
    # Find the template: a cluster of non-zero cells containing an 8 surrounded by another color
    # The 8 is the "center" and the surrounding non-8 cells form the pattern
    # Lone 8s get the pattern applied around them
    
    nonzero_vals = sorted(set(arr.flatten()) - {0})
    
    # Find all 8 positions
    eight_pos = list(zip(*np.where(arr == 8)))
    
    # Classify each 8: template (has non-8 non-0 neighbors) vs lone (no such neighbors)
    template_8 = None
    lone_8s = []
    
    for er, ec in eight_pos:
        # Check in a larger neighborhood
        has_color_neighbor = False
        for dr in range(-5, 6):
            for dc in range(-5, 6):
                nr, nc = er+dr, ec+dc
                if 0<=nr<rows and 0<=nc<cols and arr[nr,nc] not in [0,8]:
                    has_color_neighbor = True
                    break
            if has_color_neighbor:
                break
        
        if has_color_neighbor:
            template_8 = (er, ec)
        else:
            lone_8s.append((er, ec))
    
    if template_8 is None:
        return grid
    
    tr, tc = template_8
    
    # Extract pattern relative to template_8
    offsets = {}
    for val in nonzero_vals:
        if val == 8:
            continue
        positions = list(zip(*np.where(arr == val)))
        for pr, pc in positions:
            off = (pr - tr, pc - tc)
            offsets[off] = val
    
    result = arr.copy()
    
    # Remove template
    result[tr, tc] = 0
    for (dr, dc), val in offsets.items():
        nr, nc = tr+dr, tc+dc
        if 0<=nr<rows and 0<=nc<cols:
            result[nr, nc] = 0
    
    # Apply pattern to lone 8s
    for lr, lc in lone_8s:
        result[lr, lc] = 0  # Remove lone 8
        for (dr, dc), val in offsets.items():
            nr, nc = lr+dr, lc+dc
            if 0<=nr<rows and 0<=nc<cols:
                result[nr, nc] = val
    
    return result.tolist()
'''

# Now run all solutions
def run_solutions():
    log_path = RESULTS_DIR / "batch6_log.json"
    log = {}
    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)
    
    correct = 0
    total = 0
    
    all_tasks = "6e02f1e3,6e19193c,6ecd11f4,6f473927,6ffe8f07,712bf12e,72207abc,72322fa7,72ca375d,73c3b0d8,73ccf9c2,7447852a,753ea09b,758abdf0,759f3fd3,75b8110e,760b3cac,762cd429,770cc55f,776ffc46,77fdfe62,780d0b14,782b5218,7837ac64,78e78cff,79369cc6,794b24be,79cce52d,7acdf6d3,7b6016b9,7bb29440,7c008303,7c8af763,7c9b52a0,7d18a6fb,7d419a02,7d7772cc,7ddcd7ec,7df24a62,7e02026e,7e0986d6,7e2bad24,7e4d4f7c,7e576d6e,7ec998c9,7ee1c6ea,7f4411dc,80214e03,80af3007,817e6c09".split(',')
    
    for tid in all_tasks:
        total += 1
        
        if tid not in SOLUTIONS:
            if tid in log and log[tid].get('status') == 'correct':
                correct += 1
            continue
        
        code = SOLUTIONS[tid].strip()
        ok, out = verify(tid, code)
        
        if ok:
            print(f"✓ {tid}")
            log[tid] = {'status': 'correct', 'attempt': 1}
            correct += 1
        else:
            print(f"✗ {tid}: {out[:100]}")
            log[tid] = {'status': 'failed', 'output': out[:200]}
        
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2)
    
    print(f"\nSolutions applied: {sum(1 for t in SOLUTIONS if t in all_tasks)}")
    print(f"Correct: {correct}")

if __name__ == '__main__':
    run_solutions()
