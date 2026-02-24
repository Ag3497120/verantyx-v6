"""
Multi-Arm Beam Search for ARC-AGI2

Each "arm" = one cross_engine step (try all DSL primitives).
Multiple arms are chained in series, with beam search keeping top-K
intermediate results at each depth.

This is a generalization of iterative_cross_search:
- iterative_cross: 2 rounds, 1 chain, top-10 partial matches
- beam_search: N rounds, K parallel chains, scored by residual quality
"""

import time as _time
from typing import List, Tuple, Optional, Dict
from arc.grid import Grid, grid_eq, grid_shape, most_common_color


def _residual_score(piece_apply, train_pairs):
    """Score how well a piece transforms inputs toward targets.
    Returns (cell_match_ratio, n_exact_pairs).
    """
    total_cells = 0
    matching_cells = 0
    exact_pairs = 0
    
    for inp, target in train_pairs:
        result = piece_apply(inp)
        if result is None:
            return (0.0, 0)
        
        h, w = grid_shape(target)
        rh, rw = grid_shape(result)
        if (h, w) != (rh, rw):
            return (0.0, 0)
        
        pair_match = 0
        pair_total = h * w
        for r in range(h):
            for c in range(w):
                total_cells += 1
                if result[r][c] == target[r][c]:
                    matching_cells += 1
                    pair_match += 1
        
        if pair_match == pair_total:
            exact_pairs += 1
    
    ratio = matching_cells / total_cells if total_cells > 0 else 0.0
    return (ratio, exact_pairs)


def _apply_chain(chain, inp):
    """Apply a chain of pieces sequentially."""
    x = inp
    for piece in chain:
        x = piece.apply(x)
        if x is None:
            return None
    return x


def _verify_chain(chain, train_pairs):
    """Verify a chain against all training pairs."""
    for inp, target in train_pairs:
        result = _apply_chain(chain, inp)
        if result is None or not grid_eq(result, target):
            return False
    return True


def _compute_residual_pairs(chain, train_pairs):
    """Compute (intermediate_output, target) pairs after applying chain."""
    residual_pairs = []
    for inp, target in train_pairs:
        mid = _apply_chain(chain, inp)
        if mid is None:
            return None
        residual_pairs.append((mid, target))
    return residual_pairs


def beam_search_solve(train_pairs: List[Tuple[Grid, Grid]],
                      generate_pieces_fn,
                      max_depth: int = 4,
                      beam_width: int = 8,
                      time_limit: float = 5.0,
                      min_improvement: float = 0.005) -> List[Tuple[str, tuple]]:
    """
    Multi-arm beam search.
    
    At each depth:
    1. For each beam entry (chain, residual_pairs, score):
       a. Generate pieces for the current residual_pairs
       b. Score each piece by how much it improves toward target
       c. Add (chain + piece) to next beam
    2. Keep top-K entries by score
    3. If any chain achieves exact match, return immediately
    
    Args:
        train_pairs: Original (input, target) pairs
        generate_pieces_fn: Function that generates CrossPiece list from train_pairs
        max_depth: Maximum chain length
        beam_width: Number of candidates to keep at each depth
        time_limit: Total time budget in seconds
        min_improvement: Minimum score improvement to continue expanding a chain
    
    Returns:
        List of (kind, chain_tuple) for verified solutions
    """
    t0 = _time.time()
    results = []
    
    # Initial beam: empty chain, original train_pairs
    # Each entry: (score, chain, residual_pairs)
    beam = [(0.0, [], train_pairs)]
    
    # Track best score per chain to avoid cycles
    seen_signatures = set()
    
    for depth in range(max_depth):
        if _time.time() - t0 > time_limit:
            break
        
        next_beam = []
        
        for parent_score, chain, residual_pairs in beam:
            if _time.time() - t0 > time_limit:
                break
            
            # Generate pieces for current residual
            pieces = generate_pieces_fn(residual_pairs)
            
            if not pieces:
                continue
            
            # Score each piece
            scored_pieces = []
            for piece in pieces:
                # Quick check: does this piece change anything?
                score, n_exact = _residual_score(piece.apply, residual_pairs)
                
                # Must improve over parent
                if score <= parent_score + min_improvement:
                    continue
                
                # Check for exact match (all pairs)
                if n_exact == len(residual_pairs):
                    new_chain = chain + [piece]
                    # Full verification on original train_pairs
                    if _verify_chain(new_chain, train_pairs):
                        kind = f'beam_depth_{len(new_chain)}'
                        chain_names = '+'.join(p.name for p in new_chain)
                        results.append((f'{kind}({chain_names})', tuple(new_chain)))
                        return results  # Found exact match
                
                # Avoid identity/cycle: check signature
                sig_parts = [p.name for p in chain] + [piece.name]
                sig = tuple(sig_parts)
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)
                
                scored_pieces.append((score, piece))
            
            # Take top candidates
            scored_pieces.sort(key=lambda x: -x[0])
            
            for score, piece in scored_pieces[:beam_width]:
                new_chain = chain + [piece]
                new_residual = _compute_residual_pairs(new_chain, train_pairs)
                if new_residual is not None:
                    next_beam.append((score, new_chain, new_residual))
        
        if not next_beam:
            break
        
        # Keep top-K by score
        next_beam.sort(key=lambda x: -x[0])
        beam = next_beam[:beam_width]
    
    return results


def beam_search_with_residual_learners(train_pairs: List[Tuple[Grid, Grid]],
                                        generate_pieces_fn,
                                        max_depth: int = 4,
                                        beam_width: int = 8,
                                        time_limit: float = 5.0) -> List[Tuple[str, tuple]]:
    """
    Enhanced beam search that also tries residual learners at each depth.
    Residual learners are fast pattern matchers that can close small gaps.
    """
    from arc.residual_learner import ALL_LEARNERS as RES_LEARNERS
    from arc.cross_engine import CrossPiece
    
    def generate_with_residual(residual_pairs):
        pieces = generate_pieces_fn(residual_pairs)
        
        # Add residual learners as additional pieces
        for rname, rlearn, rapply in RES_LEARNERS:
            try:
                rule = rlearn(residual_pairs)
                if rule is not None:
                    _r = rule
                    _ra = rapply
                    pieces.insert(0, CrossPiece(
                        f'res:{rname}',
                        lambda inp, r=_r, fn=_ra: fn(inp, r)
                    ))
            except Exception:
                pass
        
        return pieces
    
    return beam_search_solve(
        train_pairs,
        generate_with_residual,
        max_depth=max_depth,
        beam_width=beam_width,
        time_limit=time_limit
    )
