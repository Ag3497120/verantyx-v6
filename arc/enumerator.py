"""
DSL Program Enumerator with CEGIS Verification

Systematically enumerate all programs up to depth N using DSL primitives.
Use train pair[0] as fast filter, then verify against all pairs.

Unlike beam search (greedy, monotonic), this explores ALL paths including
non-monotonic ones where intermediate steps may worsen partial match
before reaching the correct answer.
"""

import time as _time
from typing import List, Tuple, Optional, Callable
from arc.grid import Grid, grid_eq, grid_shape, most_common_color
from arc.primitives import PARAMETERLESS_PRIMITIVES, get_color_primitives


def _get_all_primitives(inp: Grid) -> List[Tuple[str, Callable]]:
    """Get all applicable primitives for a given input grid."""
    h, w = grid_shape(inp)
    prims = []
    
    for name, fn in PARAMETERLESS_PRIMITIVES:
        prims.append((name, fn))
    
    # Color-parameterized primitives
    try:
        color_prims = get_color_primitives(inp)
        prims.extend(color_prims)
    except Exception:
        pass
    
    return prims


def _filter_primitives(prims, inp, out):
    """Pre-filter: only keep primitives that produce a non-None result
    with output dimensions compatible with target."""
    oh, ow = grid_shape(out)
    ih, iw = grid_shape(inp)
    filtered = []
    
    for name, fn in prims:
        try:
            result = fn(inp)
            if result is None:
                continue
            rh, rw = grid_shape(result)
            # Don't filter by size — intermediate steps can change size
            # But skip if result == input (identity)
            if (rh, rw) == (ih, iw) and grid_eq(result, inp):
                continue
            filtered.append((name, fn, result))
        except Exception:
            continue
    
    return filtered


def enumerate_depth1(train_pairs: List[Tuple[Grid, Grid]],
                     time_limit: float = 1.0) -> List[Tuple[str, Callable]]:
    """Try all single primitives."""
    t0 = _time.time()
    inp0, out0 = train_pairs[0]
    prims = _get_all_primitives(inp0)
    results = []
    
    for name, fn in prims:
        if _time.time() - t0 > time_limit:
            break
        try:
            r0 = fn(inp0)
            if r0 is None or not grid_eq(r0, out0):
                continue
            # Verify all pairs
            ok = True
            for inp, out in train_pairs[1:]:
                r = fn(inp)
                if r is None or not grid_eq(r, out):
                    ok = False; break
            if ok:
                results.append((f'enum_d1:{name}', fn))
        except Exception:
            continue
    
    return results


def enumerate_depth2(train_pairs: List[Tuple[Grid, Grid]],
                     time_limit: float = 1.5) -> List[Tuple[str, Callable]]:
    """Try all pairs of primitives (p1 → p2).
    
    Optimization: use ONLY parameterless primitives for step 2
    (skip color-parameterized to avoid regenerating per intermediate).
    """
    t0 = _time.time()
    inp0, out0 = train_pairs[0]
    
    prims = _get_all_primitives(inp0)
    # Step 2 only uses parameterless prims (fast, no per-grid generation)
    prims2_fixed = list(PARAMETERLESS_PRIMITIVES)
    
    # Phase 1: Apply all primitives to inp0, cache intermediate results
    intermediates = []
    for name, fn in prims:
        if _time.time() - t0 > time_limit * 0.4:
            break
        try:
            mid = fn(inp0)
            if mid is None:
                continue
            ih, iw = grid_shape(inp0)
            mh, mw = grid_shape(mid)
            if (mh, mw) == (ih, iw) and grid_eq(mid, inp0):
                continue
            if mh * mw > 900:  # size guard
                continue
            intermediates.append((name, fn, mid))
        except Exception:
            continue
    
    results = []
    
    # Phase 2: For each intermediate, try fixed set of primitives
    for name1, fn1, mid0 in intermediates:
        if _time.time() - t0 > time_limit:
            break
        
        for name2, fn2 in prims2_fixed:
            if _time.time() - t0 > time_limit:
                break
            try:
                r0 = fn2(mid0)
                if r0 is None or not grid_eq(r0, out0):
                    continue
                
                # Fast match on pair[0]! Now verify all pairs
                ok = True
                for inp, out in train_pairs[1:]:
                    mid = fn1(inp)
                    if mid is None:
                        ok = False; break
                    r = fn2(mid)
                    if r is None or not grid_eq(r, out):
                        ok = False; break
                if ok:
                    combo_name = f'enum_d2:{name1}+{name2}'
                    _f1, _f2 = fn1, fn2
                    def composed(inp, f1=_f1, f2=_f2):
                        mid = f1(inp)
                        return f2(mid) if mid is not None else None
                    results.append((combo_name, composed))
                    if len(results) >= 3:
                        return results
            except Exception:
                continue
    
    return results


def enumerate_depth3(train_pairs: List[Tuple[Grid, Grid]],
                     time_limit: float = 3.0) -> List[Tuple[str, Callable]]:
    """Try all triples of primitives (p1 → p2 → p3).
    
    Uses aggressive pruning: only explore intermediate states that
    differ significantly from input and have reasonable size.
    """
    t0 = _time.time()
    inp0, out0 = train_pairs[0]
    oh, ow = grid_shape(out0)
    ih, iw = grid_shape(inp0)
    
    prims = _get_all_primitives(inp0)
    
    # Step 1: all single-step intermediates
    step1 = []
    for name, fn in prims:
        if _time.time() - t0 > time_limit * 0.3:
            break
        try:
            mid = fn(inp0)
            if mid is None:
                continue
            mh, mw = grid_shape(mid)
            if (mh, mw) == (ih, iw) and grid_eq(mid, inp0):
                continue
            # Size guard: don't let grid grow too large
            if mh * mw > 900:
                continue
            step1.append((name, fn, mid))
        except Exception:
            continue
    
    results = []
    
    # Step 2: for each step1 result, apply another primitive
    step2 = []
    for name1, fn1, mid1 in step1:
        if _time.time() - t0 > time_limit * 0.6:
            break
        
        prims2 = _get_all_primitives(mid1)
        for name2, fn2 in prims2:
            if _time.time() - t0 > time_limit * 0.6:
                break
            try:
                mid2 = fn2(mid1)
                if mid2 is None:
                    continue
                m2h, m2w = grid_shape(mid2)
                mh, mw = grid_shape(mid1)
                if (m2h, m2w) == (mh, mw) and grid_eq(mid2, mid1):
                    continue
                if m2h * m2w > 900:
                    continue
                step2.append((name1, fn1, name2, fn2, mid2))
            except Exception:
                continue
    
    # Step 3: for each step2 result, try final primitive
    for name1, fn1, name2, fn2, mid2 in step2:
        if _time.time() - t0 > time_limit:
            break
        
        prims3 = _get_all_primitives(mid2)
        for name3, fn3 in prims3:
            if _time.time() - t0 > time_limit:
                break
            try:
                r0 = fn3(mid2)
                if r0 is None or not grid_eq(r0, out0):
                    continue
                
                # Match pair[0]! Verify all
                ok = True
                for inp, out in train_pairs[1:]:
                    m1 = fn1(inp)
                    if m1 is None: ok = False; break
                    m2 = fn2(m1)
                    if m2 is None: ok = False; break
                    r = fn3(m2)
                    if r is None or not grid_eq(r, out): ok = False; break
                if ok:
                    combo_name = f'enum_d3:{name1}+{name2}+{name3}'
                    _f1, _f2, _f3 = fn1, fn2, fn3
                    def composed(inp, f1=_f1, f2=_f2, f3=_f3):
                        m1 = f1(inp)
                        if m1 is None: return None
                        m2 = f2(m1)
                        if m2 is None: return None
                        return f3(m2)
                    results.append((combo_name, composed))
                    if len(results) >= 3:
                        return results
            except Exception:
                continue
    
    return results


def enumerate_solve(train_pairs: List[Tuple[Grid, Grid]],
                    max_depth: int = 3,
                    time_limit: float = 4.0) -> List[Tuple[str, Callable]]:
    """Main entry point: enumerate programs up to max_depth.
    
    Returns list of (name, apply_fn) for verified solutions.
    """
    t0 = _time.time()
    remaining = time_limit
    
    # Depth 1
    results = enumerate_depth1(train_pairs, time_limit=min(0.5, remaining))
    if results:
        return results
    remaining = time_limit - (_time.time() - t0)
    if remaining <= 0:
        return []
    
    # Depth 2
    if max_depth >= 2:
        results = enumerate_depth2(train_pairs, time_limit=min(1.5, remaining))
        if results:
            return results
        remaining = time_limit - (_time.time() - t0)
        if remaining <= 0:
            return []
    
    # Depth 3
    if max_depth >= 3:
        results = enumerate_depth3(train_pairs, time_limit=min(remaining, 3.0))
        if results:
            return results
    
    return []
