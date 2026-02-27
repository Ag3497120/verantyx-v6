"""Actually try to solve gravity tasks using probe measurements"""
import json
import numpy as np
from arc.cross3d_probe import measure_objects, learn_transform_from_probes

with open('llm_classifications.json') as f:
    cls = json.load(f)

gravity_tasks = [tid for tid, info in cls.items() if info.get('primary') == 'gravity']


def try_solve_gravity(task):
    """Try to solve using probe-based object tracking"""
    train_pairs = [(p['input'], p['output']) for p in task['train']]
    test_pairs = [(p['input'], p.get('output', p['input'])) for p in task['test']]
    
    # Strategy 1: Learn per-object movement and generalize
    # For each train pair, measure objects and their movements
    
    # First, check if movement pattern is consistent across train examples
    all_pair_data = []
    
    for inp_grid, out_grid in train_pairs:
        inp_np = np.array(inp_grid, dtype=int)
        out_np = np.array(out_grid, dtype=int)
        
        inp_objs = measure_objects(inp_np, bg=0)
        out_objs = measure_objects(out_np, bg=0)
        
        # Match by color + closest position
        movements = []
        used = set()
        
        for io in sorted(inp_objs, key=lambda o: -o['area']):
            best_j = None
            best_dist = 999
            for j, oo in enumerate(out_objs):
                if j in used or io['color'] != oo['color']:
                    continue
                dist = abs(io['area'] - oo['area'])
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
            if best_j is not None:
                used.add(best_j)
                oo = out_objs[best_j]
                movements.append({
                    'color': io['color'],
                    'inp_bbox': io['bbox'],
                    'out_bbox': oo['bbox'],
                    'delta_r': oo['bbox'][0] - io['bbox'][0],
                    'delta_c': oo['bbox'][1] - io['bbox'][1],
                    'area': io['area'],
                    'inp_center': io['center'],
                    'out_center': oo['center'],
                })
        
        all_pair_data.append({
            'inp_np': inp_np,
            'out_np': out_np,
            'inp_objs': inp_objs,
            'out_objs': out_objs,
            'movements': movements,
        })
    
    # Strategy A: Uniform gravity (all objects move same direction, stack at wall)
    result_a = try_uniform_gravity(all_pair_data, train_pairs)
    if result_a is not None:
        return result_a, 'uniform_gravity'
    
    # Strategy B: Diagonal stacking (objects move toward corner, stack on each other)
    result_b = try_diagonal_stack(all_pair_data, train_pairs)
    if result_b is not None:
        return result_b, 'diagonal_stack'
    
    # Strategy C: Per-color movement
    result_c = try_per_color(all_pair_data, train_pairs)
    if result_c is not None:
        return result_c, 'per_color'
    
    return None, 'failed'


def try_uniform_gravity(all_pair_data, train_pairs):
    """All objects slide in the same direction until hitting wall or other object"""
    for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
        all_match = True
        for pd in all_pair_data:
            result = apply_gravity(pd['inp_np'], dr, dc)
            if not np.array_equal(result, pd['out_np']):
                all_match = False
                break
        if all_match:
            return (dr, dc)
    return None


def try_diagonal_stack(all_pair_data, train_pairs):
    """Objects stack diagonally - order by position, each slides until touching previous"""
    for dr, dc in [(-1,-1), (-1,1), (1,-1), (1,1)]:
        all_match = True
        for pd in all_pair_data:
            result = apply_diagonal_gravity(pd['inp_np'], dr, dc)
            if not np.array_equal(result, pd['out_np']):
                all_match = False
                break
        if all_match:
            return (dr, dc)
    return None


def try_per_color(all_pair_data, train_pairs):
    """Each color has its own movement direction"""
    # Learn direction per color from first pair
    if not all_pair_data or not all_pair_data[0]['movements']:
        return None
    
    color_dirs = {}
    for m in all_pair_data[0]['movements']:
        dr, dc = m['delta_r'], m['delta_c']
        if dr == 0 and dc == 0:
            continue
        # Normalize to unit direction
        if abs(dr) > abs(dc):
            ndr = 1 if dr > 0 else -1
            ndc = 0
        else:
            ndr = 0
            ndc = 1 if dc > 0 else -1
        color_dirs[m['color']] = (ndr, ndc)
    
    if not color_dirs:
        return None
    
    # Verify on all train pairs
    all_match = True
    for pd in all_pair_data:
        result = apply_per_color_gravity(pd['inp_np'], color_dirs)
        if not np.array_equal(result, pd['out_np']):
            all_match = False
            break
    
    if all_match:
        return color_dirs
    return None


def apply_gravity(grid, dr, dc):
    """Apply uniform gravity: all non-bg objects slide in direction (dr,dc)"""
    H, W = grid.shape
    result = np.zeros_like(grid)
    
    objs = measure_objects(grid, bg=0)
    
    # Sort: objects furthest in gravity direction move first (they're already at destination)
    if dr > 0:
        objs.sort(key=lambda o: -o['bbox'][2])  # bottom first
    elif dr < 0:
        objs.sort(key=lambda o: o['bbox'][0])   # top first
    elif dc > 0:
        objs.sort(key=lambda o: -o['bbox'][3])  # right first
    elif dc < 0:
        objs.sort(key=lambda o: o['bbox'][1])   # left first
    
    placed = np.zeros_like(grid, dtype=bool)
    
    for obj in objs:
        r_min, c_min, r_max, c_max = obj['bbox']
        obj_h = r_max - r_min + 1
        obj_w = c_max - c_min + 1
        
        # Extract object mask relative to bbox
        mask = np.zeros((obj_h, obj_w), dtype=bool)
        for cr, cc in obj['cells']:
            mask[cr - r_min, cc - c_min] = True
        
        # Slide until hitting boundary or placed object
        cur_r, cur_c = r_min, c_min
        while True:
            next_r = cur_r + dr
            next_c = cur_c + dc
            
            # Check bounds
            if next_r < 0 or next_c < 0 or next_r + obj_h > H or next_c + obj_w > W:
                break
            
            # Check collision with placed objects
            collision = False
            for mr in range(obj_h):
                for mc in range(obj_w):
                    if mask[mr, mc]:
                        if placed[next_r + mr, next_c + mc]:
                            collision = True
                            break
                if collision:
                    break
            
            if collision:
                break
            
            cur_r, cur_c = next_r, next_c
        
        # Place object
        for mr in range(obj_h):
            for mc in range(obj_w):
                if mask[mr, mc]:
                    result[cur_r + mr, cur_c + mc] = obj['color']
                    placed[cur_r + mr, cur_c + mc] = True
    
    return result


def apply_diagonal_gravity(grid, dr, dc):
    """
    Diagonal stacking: objects slide toward a corner.
    Sort by distance from target corner (closest first = arrives first = stack base).
    """
    H, W = grid.shape
    result = np.zeros_like(grid)
    
    objs = measure_objects(grid, bg=0)
    
    # Target corner
    target_r = 0 if dr < 0 else H - 1
    target_c = 0 if dc < 0 else W - 1
    
    # Sort: closest to target corner first (they form the base)
    objs.sort(key=lambda o: abs(o['center'][0] - target_r) + abs(o['center'][1] - target_c))
    
    placed = np.zeros_like(grid, dtype=bool)
    
    for obj in objs:
        r_min, c_min, r_max, c_max = obj['bbox']
        obj_h = r_max - r_min + 1
        obj_w = c_max - c_min + 1
        
        mask = np.zeros((obj_h, obj_w), dtype=bool)
        for cr, cc in obj['cells']:
            mask[cr - r_min, cc - c_min] = True
        
        # Slide diagonally
        cur_r, cur_c = r_min, c_min
        
        while True:
            moved = False
            
            # Try diagonal first
            next_r, next_c = cur_r + dr, cur_c + dc
            if can_place(next_r, next_c, obj_h, obj_w, mask, placed, H, W):
                cur_r, cur_c = next_r, next_c
                moved = True
            else:
                # Try each axis separately
                nr1, nc1 = cur_r + dr, cur_c
                nr2, nc2 = cur_r, cur_c + dc
                
                can1 = can_place(nr1, nc1, obj_h, obj_w, mask, placed, H, W)
                can2 = can_place(nr2, nc2, obj_h, obj_w, mask, placed, H, W)
                
                if can1 and can2:
                    # Prefer the axis that moves closer to target
                    d1 = abs(nr1 + obj_h//2 - target_r) + abs(nc1 + obj_w//2 - target_c)
                    d2 = abs(nr2 + obj_h//2 - target_r) + abs(nc2 + obj_w//2 - target_c)
                    if d1 <= d2:
                        cur_r, cur_c = nr1, nc1
                    else:
                        cur_r, cur_c = nr2, nc2
                    moved = True
                elif can1:
                    cur_r, cur_c = nr1, nc1
                    moved = True
                elif can2:
                    cur_r, cur_c = nr2, nc2
                    moved = True
            
            if not moved:
                break
        
        # Place
        for mr in range(obj_h):
            for mc in range(obj_w):
                if mask[mr, mc]:
                    result[cur_r + mr, cur_c + mc] = obj['color']
                    placed[cur_r + mr, cur_c + mc] = True
    
    return result


def apply_per_color_gravity(grid, color_dirs):
    """Each color slides in its own direction"""
    H, W = grid.shape
    result = np.zeros_like(grid)
    placed = np.zeros_like(grid, dtype=bool)
    
    objs = measure_objects(grid, bg=0)
    
    # Place non-moving objects first
    for obj in objs:
        if obj['color'] not in color_dirs:
            for cr, cc in obj['cells']:
                result[cr, cc] = obj['color']
                placed[cr, cc] = True
    
    # Then slide moving objects
    for obj in objs:
        if obj['color'] not in color_dirs:
            continue
        
        dr, dc = color_dirs[obj['color']]
        r_min, c_min, r_max, c_max = obj['bbox']
        obj_h = r_max - r_min + 1
        obj_w = c_max - c_min + 1
        
        mask = np.zeros((obj_h, obj_w), dtype=bool)
        for cr, cc in obj['cells']:
            mask[cr - r_min, cc - c_min] = True
        
        cur_r, cur_c = r_min, c_min
        while True:
            next_r, next_c = cur_r + dr, cur_c + dc
            if can_place(next_r, next_c, obj_h, obj_w, mask, placed, H, W):
                cur_r, cur_c = next_r, next_c
            else:
                break
        
        for mr in range(obj_h):
            for mc in range(obj_w):
                if mask[mr, mc]:
                    result[cur_r + mr, cur_c + mc] = obj['color']
                    placed[cur_r + mr, cur_c + mc] = True
    
    return result


def can_place(r, c, h, w, mask, placed, H, W):
    """Check if object can be placed at (r,c) without collision"""
    if r < 0 or c < 0 or r + h > H or c + w > W:
        return False
    for mr in range(h):
        for mc in range(w):
            if mask[mr, mc] and placed[r + mr, c + mc]:
                return False
    return True


# ─── Run tests ───
solved = 0
strategy_counts = {}
fail_details = []

for tid in sorted(gravity_tasks):
    with open(f'/tmp/arc-agi-2/data/training/{tid}.json') as f:
        task = json.load(f)
    
    result, strategy = try_solve_gravity(task)
    
    # Verify on ALL train examples
    verified = False
    if result is not None:
        verified = True
        for pair in task['train']:
            inp_np = np.array(pair['input'])
            out_np = np.array(pair['output'])
            
            if strategy == 'uniform_gravity':
                dr, dc = result
                pred = apply_gravity(inp_np, dr, dc)
            elif strategy == 'diagonal_stack':
                dr, dc = result
                pred = apply_diagonal_gravity(inp_np, dr, dc)
            elif strategy == 'per_color':
                pred = apply_per_color_gravity(inp_np, result)
            else:
                verified = False
                break
            
            if not np.array_equal(pred, out_np):
                verified = False
                break
    
    if verified:
        solved += 1
        mark = '✅'
    else:
        mark = '❌'
        # Show diff for first train pair
        inp_np = np.array(task['train'][0]['input'])
        out_np = np.array(task['train'][0]['output'])
        fail_details.append((tid, strategy))
    
    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    print(f'{mark} {tid}: {strategy}')

print(f'\n=== Results ===')
print(f'Solved (all train verified): {solved}/{len(gravity_tasks)}')
print(f'\nStrategy distribution:')
for s, c in sorted(strategy_counts.items(), key=lambda x: -x[1]):
    print(f'  {s}: {c}')
print(f'\nFailed tasks:')
for tid, strat in fail_details:
    print(f'  {tid}: {strat}')
