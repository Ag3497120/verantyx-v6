"""
arc/cross_vote.py — 投票制ソルバー: 足し算 × 引き算 のハイブリッド

1. 足し算ソルバー群 (brute, cross6_ops, fill_v2, cell_v1, etc.)
2. 引き算ソルバー (cross_cut)
3. 両方の結果を集めて投票

投票ルール:
  - 複数ソルバーが同じ答え → 信頼度最高
  - 1つだけ → そのまま返す
  - 矛盾 → 多数決、同数なら引き算優先（汎化しやすい）
"""

from typing import List, Tuple, Optional
from collections import Counter
from arc.grid import grid_eq
import json


def cross_vote_solve(train_pairs, test_input) -> Optional[List[List[int]]]:
    """投票制ソルバー"""
    candidates = []

    # ─── 足し算ソルバー群 ───
    additive_solvers = _get_additive_solvers()
    for name, solver in additive_solvers:
        try:
            r = solver(train_pairs, test_input)
            if r is not None:
                candidates.append((name, r))
        except Exception:
            continue

    # ─── 引き算ソルバー ───
    try:
        from arc.cross_cut import cross_cut_solve
        r = cross_cut_solve(train_pairs, test_input)
        if r is not None:
            candidates.append(('cross_cut', r))
    except Exception:
        pass

    if not candidates:
        return None

    # ─── 投票 ───
    if len(candidates) == 1:
        return candidates[0][1]

    # グリッドをキーにグループ化
    groups = {}
    for name, result in candidates:
        key = json.dumps(result)
        if key not in groups:
            groups[key] = {'result': result, 'voters': [], 'count': 0}
        groups[key]['voters'].append(name)
        groups[key]['count'] += 1

    # 多数決
    best = max(groups.values(), key=lambda g: (
        g['count'],
        # 同数なら引き算を優先
        1 if 'cross_cut' in g['voters'] else 0,
        # 次にbrute
        1 if 'brute' in g['voters'] else 0,
    ))

    return best['result']


def _get_additive_solvers():
    """足し算ソルバー一覧"""
    solvers = []

    try:
        from arc.cross6_brute import brute_solve
        solvers.append(('brute', brute_solve))
    except ImportError:
        pass

    try:
        from arc.cross_world import cross_world_solve
        solvers.append(('cross_world', cross_world_solve))
    except ImportError:
        pass

    try:
        from arc.cross6_ops import cross6_ops_solve
        solvers.append(('cross6_ops', cross6_ops_solve))
    except ImportError:
        pass

    try:
        from arc.cross6_fill_v2 import cross6_fill_v2_solve
        solvers.append(('fill_v2', cross6_fill_v2_solve))
    except ImportError:
        pass

    try:
        from arc.cross6axis import cross6_solve
        solvers.append(('cell_v1', cross6_solve))
    except ImportError:
        pass

    try:
        from arc.cross6_transform import global_transform_solve, color_swap_solve
        solvers.append(('global_transform', global_transform_solve))
        solvers.append(('color_swap', color_swap_solve))
    except ImportError:
        pass

    try:
        from arc.cross6_scale import size_change_solve
        solvers.append(('size_change', size_change_solve))
    except ImportError:
        pass

    return solvers


# ──── CLI ────

if __name__ == "__main__":
    import sys
    import re
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python3 -m arc.cross_vote [--eval | --train | task.json]")
        sys.exit(1)

    if sys.argv[1] == '--eval':
        EVAL_DIR = Path('/tmp/arc-agi-2/data/evaluation')
        solved = []
        for tf in sorted(EVAL_DIR.glob('*.json')):
            tid = tf.stem
            with open(tf) as f:
                task = json.load(f)
            tp = [(e['input'], e['output']) for e in task['train']]
            ti, to = task['test'][0]['input'], task['test'][0].get('output')
            r = cross_vote_solve(tp, ti)
            if r and to and grid_eq(r, to):
                solved.append(tid)
                print(f'✅ {tid}')
        print(f'\nEval: {len(solved)}/120')

    elif sys.argv[1] == '--train':
        TRAIN_DIR = Path('/tmp/arc-agi-2/data/training')
        existing = set()
        try:
            with open('arc_cross_engine_v9.log') as f:
                for line in f:
                    m = re.search(r'✓.*?([0-9a-f]{8})', line)
                    if m:
                        existing.add(m.group(1))
        except:
            pass

        solved = []
        for tf in sorted(TRAIN_DIR.glob('*.json')):
            tid = tf.stem
            with open(tf) as f:
                task = json.load(f)
            tp = [(e['input'], e['output']) for e in task['train']]
            ti, to = task['test'][0]['input'], task['test'][0].get('output')
            r = cross_vote_solve(tp, ti)
            if r and to and grid_eq(r, to):
                solved.append(tid)

        new = [t for t in solved if t not in existing]
        print(f'Training: {len(solved)}/1000 (NEW: {len(new)})')
        for t in sorted(new):
            print(f'  NEW: {t}')

    else:
        with open(sys.argv[1]) as f:
            task = json.load(f)
        tp = [(e['input'], e['output']) for e in task['train']]
        ti = task['test'][0]['input']
        to = task['test'][0].get('output')
        r = cross_vote_solve(tp, ti)
        if r:
            print('✅ SOLVED!' if to and grid_eq(r, to) else '⚠️ Solution')
        else:
            print('✗ No solution')
