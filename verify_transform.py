#!/usr/bin/env python3
"""Verify a transform function against task train/test examples."""
import json, sys, numpy as np, threading
from collections import Counter, defaultdict

def verify(task_path, code_str):
    with open(task_path) as f:
        task = json.load(f)
    
    ns = {'np': np, 'numpy': np, 'Counter': Counter, 'defaultdict': defaultdict}
    try:
        import scipy; ns['scipy'] = scipy
        from scipy.ndimage import label; ns['label'] = label
    except: pass
    
    try:
        exec(code_str, ns)
    except Exception as e:
        return {'status': 'exec_error', 'error': str(e)}
    
    if 'transform' not in ns:
        return {'status': 'no_transform'}
    
    fn = ns['transform']
    
    # Verify train
    for i, ex in enumerate(task['train']):
        result = [None]
        def run(): 
            try: result[0] = fn(ex['input'])
            except: pass
        t = threading.Thread(target=run); t.start(); t.join(5)
        if result[0] is None:
            return {'status': 'train_fail', 'index': i, 'reason': 'crash_or_timeout'}
        pred = [[int(c) for c in row] for row in result[0]]
        if not np.array_equal(np.array(pred), np.array(ex['output'])):
            return {'status': 'train_fail', 'index': i, 'reason': 'mismatch'}
    
    # Verify test
    test_outputs = []
    for i, ex in enumerate(task['test']):
        result = [None]
        def run():
            try: result[0] = fn(ex['input'])
            except: pass
        t = threading.Thread(target=run); t.start(); t.join(5)
        if result[0] is None:
            return {'status': 'test_fail', 'index': i}
        pred = [[int(c) for c in row] for row in result[0]]
        test_outputs.append(pred)
        if not np.array_equal(np.array(pred), np.array(ex['output'])):
            return {'status': 'test_wrong', 'index': i}
    
    return {'status': 'correct', 'test_outputs': test_outputs}

if __name__ == '__main__':
    task_path = sys.argv[1]
    code_file = sys.argv[2]
    with open(code_file) as f:
        code = f.read()
    result = verify(task_path, code)
    print(json.dumps(result))
