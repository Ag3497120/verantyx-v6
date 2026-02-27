"""Test cross3d_probe on all 51 gravity tasks"""
import json
import numpy as np
from arc.cross3d_probe import measure_objects, learn_transform_from_probes

with open('llm_classifications.json') as f:
    cls = json.load(f)

gravity_tasks = [tid for tid, info in cls.items() if info.get('primary') == 'gravity']

# Also check current eval results
try:
    with open('arc_v62_full.log') as f:
        log = f.read()
except:
    log = ''

results = {'solved': 0, 'transform_found': 0, 'total': len(gravity_tasks)}
details = []

for tid in sorted(gravity_tasks):
    with open(f'/tmp/arc-agi-2/data/training/{tid}.json') as f:
        task = json.load(f)
    
    train_pairs = [(p['input'], p['output']) for p in task['train']]
    
    # Check sizes
    same_size = all(
        np.array(p['input']).shape == np.array(p['output']).shape 
        for p in task['train']
    )
    
    # Get object counts
    inp0 = np.array(task['train'][0]['input'])
    out0 = np.array(task['train'][0]['output'])
    
    inp_objs = measure_objects(inp0, bg=0)
    out_objs = measure_objects(out0, bg=0)
    
    # Learn transform
    transform = learn_transform_from_probes(train_pairs, bg=0)
    
    has_transform = transform is not None and len(transform['train_movements']) > 0 and len(transform['train_movements'][0]) > 0
    
    if has_transform:
        results['transform_found'] += 1
    
    # Analyze movement pattern
    pattern = 'none'
    if has_transform:
        movs = transform['train_movements'][0]
        deltas = [m['delta'] for m in movs]
        
        # Check if all same direction
        if len(set(deltas)) == 1:
            pattern = f'uniform({deltas[0]})'
        else:
            # Check if diagonal stacking
            drs = [d[0] for d in deltas]
            dcs = [d[1] for d in deltas]
            if all(dr < 0 for dr in drs if dr != 0) or all(dr > 0 for dr in drs if dr != 0):
                if all(dc < 0 for dc in dcs if dc != 0) or all(dc > 0 for dc in dcs if dc != 0):
                    pattern = f'diagonal_stack'
                else:
                    pattern = f'vertical_stack'
            elif all(dc < 0 for dc in dcs if dc != 0) or all(dc > 0 for dc in dcs if dc != 0):
                pattern = f'horizontal_stack'
            else:
                pattern = f'mixed'
    
    # Check if already solved in v62
    already_solved = f'"{tid}"' in log and 'ver=5' in log  # rough check
    
    detail = {
        'tid': tid,
        'same_size': same_size,
        'n_train': len(task['train']),
        'inp_shape': inp0.shape,
        'inp_objs': len(inp_objs),
        'out_objs': len(out_objs),
        'has_transform': has_transform,
        'pattern': pattern,
        'n_movements': len(transform['train_movements'][0]) if has_transform else 0,
    }
    
    if has_transform:
        detail['movements'] = transform['train_movements'][0]
    
    details.append(detail)
    
    status = '✓' if has_transform else '✗'
    print(f'{status} {tid}: inp_objs={len(inp_objs):2d} out_objs={len(out_objs):2d} pattern={pattern:20s} same_size={same_size}')

print(f'\n=== Summary ===')
print(f'Total: {results["total"]}')
print(f'Transform found: {results["transform_found"]}')

# Pattern distribution
from collections import Counter
patterns = Counter(d['pattern'] for d in details)
print(f'\nPattern distribution:')
for p, c in patterns.most_common():
    print(f'  {p}: {c}')

# Same size distribution
same = sum(1 for d in details if d['same_size'])
diff = sum(1 for d in details if not d['same_size'])
print(f'\nSame size: {same}, Different size: {diff}')
