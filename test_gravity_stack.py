"""Test corner-stacking gravity on all 51 tasks"""
import json
import numpy as np
from arc.cross3d_probe import measure_objects

with open('llm_classifications.json') as f:
    cls = json.load(f)

gravity_tasks = [tid for tid, info in cls.items() if info.get('primary') == 'gravity']


def place_mask(r, c, mask, color, result, placed):
    mh, mw = mask.shape
    for mr in range(mh):
        for mc in range(mw):
            if mask[mr, mc]:
                result[r + mr, c + mc] = color
                placed[r + mr, c + mc] = True


def try_corner_stack(task, corner):
    """
    Stack objects toward a corner.
    corner: (tr, tc) where tr=0 means top, tr=1 means bottom; tc similar
    """
    train_pairs = task['train']
    
    for pair in train_pairs:
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        H, W = inp.shape
        
        if inp.shape != out.shape:
            return False
        
        objs = measure_objects(inp, bg=0)
        if not objs:
            return False
        
        # Target corner position
        target_r = 0 if corner[0] == 0 else H - 1
        target_c = 0 if corner[1] == 0 else W - 1
        
        # Sort by distance to target corner
        # dr_sign: -1 if moving up (target_r=0), +1 if moving down
        dr_sign = -1 if corner[0] == 0 else 1
        dc_sign = -1 if corner[1] == 0 else 1
        
        objs.sort(key=lambda o: abs(o['center'][0] - target_r) + abs(o['center'][1] - target_c))
        
        result = np.zeros_like(inp)
        placed = np.zeros_like(inp, dtype=bool)
        
        # Stack point starts at target corner
        if corner[0] == 0 and corner[1] == 0:
            stack_r, stack_c = 0, 0
        elif corner[0] == 0 and corner[1] == 1:
            stack_r = 0
            # Will adjust per object
            first_w = objs[0]['bbox'][3] - objs[0]['bbox'][1] + 1 if objs else 0
            stack_c = W - first_w
        elif corner[0] == 1 and corner[1] == 0:
            first_h = objs[0]['bbox'][2] - objs[0]['bbox'][0] + 1 if objs else 0
            stack_r = H - first_h
            stack_c = 0
        else:  # (1,1)
            first_h = objs[0]['bbox'][2] - objs[0]['bbox'][0] + 1 if objs else 0
            first_w = objs[0]['bbox'][3] - objs[0]['bbox'][1] + 1 if objs else 0
            stack_r = H - first_h
            stack_c = W - first_w
        
        for idx, obj in enumerate(objs):
            r_min, c_min, r_max, c_max = obj['bbox']
            obj_h = r_max - r_min + 1
            obj_w = c_max - c_min + 1
            
            mask = np.zeros((obj_h, obj_w), dtype=bool)
            for cr, cc in obj['cells']:
                mask[cr - r_min, cc - c_min] = True
            
            if idx == 0:
                cur_r, cur_c = stack_r, stack_c
            else:
                # Place at stack point
                if corner == (0, 0):
                    cur_r, cur_c = stack_r, stack_c
                elif corner == (0, 1):
                    cur_r = stack_r
                    cur_c = stack_c - obj_w + 1
                elif corner == (1, 0):
                    cur_r = stack_r - obj_h + 1
                    cur_c = stack_c
                else:  # (1, 1)
                    cur_r = stack_r - obj_h + 1
                    cur_c = stack_c - obj_w + 1
            
            # Bounds check
            if cur_r < 0 or cur_c < 0 or cur_r + obj_h > H or cur_c + obj_w > W:
                return False
            
            place_mask(cur_r, cur_c, mask, obj['color'], result, placed)
            
            # Update stack point
            if corner == (0, 0):
                stack_r = cur_r + obj_h - 1
                stack_c = cur_c + obj_w - 1
            elif corner == (0, 1):
                stack_r = cur_r + obj_h - 1
                stack_c = cur_c
            elif corner == (1, 0):
                stack_r = cur_r
                stack_c = cur_c + obj_w - 1
            else:  # (1, 1)
                stack_r = cur_r
                stack_c = cur_c
        
        if not np.array_equal(result, out):
            return False
    
    return True


def try_uniform_gravity(task):
    """All objects slide in same direction until wall/collision"""
    train_pairs = task['train']
    
    for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
        all_match = True
        for pair in train_pairs:
            inp = np.array(pair['input'])
            out = np.array(pair['output'])
            if inp.shape != out.shape:
                all_match = False
                break
            
            H, W = inp.shape
            objs = measure_objects(inp, bg=0)
            
            # Sort: furthest in gravity direction first
            if dr > 0: objs.sort(key=lambda o: -o['bbox'][2])
            elif dr < 0: objs.sort(key=lambda o: o['bbox'][0])
            elif dc > 0: objs.sort(key=lambda o: -o['bbox'][3])
            else: objs.sort(key=lambda o: o['bbox'][1])
            
            result = np.zeros_like(inp)
            placed = np.zeros_like(inp, dtype=bool)
            
            for obj in objs:
                r_min, c_min, r_max, c_max = obj['bbox']
                obj_h = r_max - r_min + 1
                obj_w = c_max - c_min + 1
                mask = np.zeros((obj_h, obj_w), dtype=bool)
                for cr, cc in obj['cells']:
                    mask[cr - r_min, cc - c_min] = True
                
                cur_r, cur_c = r_min, c_min
                while True:
                    nr, nc = cur_r + dr, cur_c + dc
                    if nr < 0 or nc < 0 or nr + obj_h > H or nc + obj_w > W:
                        break
                    ok = True
                    for mr in range(obj_h):
                        for mc in range(obj_w):
                            if mask[mr, mc] and placed[nr + mr, nc + mc]:
                                ok = False
                                break
                        if not ok: break
                    if not ok: break
                    cur_r, cur_c = nr, nc
                
                place_mask(cur_r, cur_c, mask, obj['color'], result, placed)
            
            if not np.array_equal(result, out):
                all_match = False
                break
        
        if all_match:
            return (dr, dc)
    
    return None


solved = 0
strategies = {}

for tid in sorted(gravity_tasks):
    with open(f'/tmp/arc-agi-2/data/training/{tid}.json') as f:
        task = json.load(f)
    
    found = False
    
    # Try corner stacking (4 corners)
    for corner in [(0,0), (0,1), (1,0), (1,1)]:
        if try_corner_stack(task, corner):
            solved += 1
            strat = f'corner_stack{corner}'
            strategies[strat] = strategies.get(strat, 0) + 1
            print(f'✅ {tid}: {strat}')
            found = True
            break
    
    if not found:
        # Try uniform gravity
        result = try_uniform_gravity(task)
        if result is not None:
            solved += 1
            strat = f'uniform{result}'
            strategies[strat] = strategies.get(strat, 0) + 1
            print(f'✅ {tid}: {strat}')
            found = True
    
    if not found:
        print(f'❌ {tid}')
        strategies['failed'] = strategies.get('failed', 0) + 1

print(f'\n=== Results ===')
print(f'Solved: {solved}/{len(gravity_tasks)}')
print(f'\nStrategies:')
for s, c in sorted(strategies.items(), key=lambda x: -x[1]):
    print(f'  {s}: {c}')
