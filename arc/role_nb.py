"""
arc/role_nb.py — Role-aware Neighborhood Rule Learner

Uses ObjectIR to add object-level context to NB patterns.
This allows discrimination between cells with identical local neighborhoods
but different structural roles (e.g., object interior vs border vs hole).

Multiple abstraction levels are tried:
1. NB + full role signature (most specific)
2. NB + compact role (balanced)  
3. Compact role only (most general)
4. NB_canonical + compact role (rotation-invariant + role)
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from collections import Counter
from arc.grid import Grid, grid_shape, most_common_color, grid_eq, grid_colors
from arc.object_ir import ObjectIR, build_object_ir


def _get_output_role(out_val, center, bg, local_other_map=None):
    """Abstract output to role string."""
    if out_val == bg:
        return 'bg'
    elif out_val == center:
        return 'keep'
    else:
        return f'abs_{out_val}'


def _resolve_role(role, center, bg):
    """Resolve role string to concrete color."""
    if role == 'bg':
        return bg
    elif role == 'keep':
        return center
    elif role.startswith('abs_'):
        return int(role.split('_')[1])
    return center


def _rotate_3x3(pat):
    m = [list(pat[0:3]), list(pat[3:6]), list(pat[6:9])]
    r = [[m[2][0], m[1][0], m[0][0]],
         [m[2][1], m[1][1], m[0][1]],
         [m[2][2], m[1][2], m[0][2]]]
    return tuple(r[0] + r[1] + r[2])

def _flip_h_3x3(pat):
    m = [list(pat[0:3]), list(pat[3:6]), list(pat[6:9])]
    return tuple(m[0][::-1] + m[1][::-1] + m[2][::-1])

def _canonical_3x3(pat):
    variants = []
    p = pat
    for _ in range(4):
        variants.append(p)
        variants.append(_flip_h_3x3(p))
        p = _rotate_3x3(p)
    return min(variants)


def _learn_with_keyfn(train_pairs, bg, key_fn_name):
    """Generic learner: try a specific key function, return mapping if consistent."""
    mapping = {}
    
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        ir = build_object_ir(inp, bg)
        
        for r in range(h):
            for c in range(w):
                center = inp[r][c]
                out_val = out[r][c]
                
                # Get key based on strategy
                if key_fn_name == 'nb_plus_role':
                    key = ir.get_nb_plus_role(r, c)
                elif key_fn_name == 'nb_plus_compact':
                    nb_part, role_part = ir.get_nb_plus_role(r, c)
                    compact = ir.get_compact_role(r, c)
                    key = (nb_part, compact)
                elif key_fn_name == 'compact_only':
                    key = ir.get_compact_role(r, c)
                elif key_fn_name == 'nb_canonical_plus_compact':
                    nb_part, role_part = ir.get_nb_plus_role(r, c)
                    cnb = _canonical_3x3(nb_part)
                    compact = ir.get_compact_role(r, c)
                    key = (cnb, compact)
                elif key_fn_name == 'signature':
                    key = ir.get_role_signature(r, c)
                elif key_fn_name == 'nb_plus_signature':
                    nb_part, _ = ir.get_nb_plus_role(r, c)
                    sig = ir.get_role_signature(r, c)
                    key = (nb_part, sig)
                else:
                    return None
                
                role = _get_output_role(out_val, center, bg)
                
                if key in mapping:
                    if mapping[key] != role:
                        return None  # inconsistent
                else:
                    mapping[key] = role
    
    # Check non-trivial
    if not mapping:
        return None
    roles = set(mapping.values())
    if roles == {'keep'}:
        return None
    
    return mapping


def learn_role_nb_rule(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Try multiple abstraction levels and return the best (most general) consistent one."""
    bg = most_common_color(train_pairs[0][0])
    
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    if grid_eq(train_pairs[0][0], train_pairs[0][1]):
        return None
    
    # Try from most general to most specific
    # More general = better generalization to test
    strategies = [
        'compact_only',           # Most general: only role features
        'nb_canonical_plus_compact',  # Canonical NB + compact role
        'nb_plus_compact',        # Full NB + compact role
        'signature',              # Full role signature (no NB)
        'nb_plus_role',           # NB + full role
        'nb_plus_signature',      # NB + full signature (most specific)
    ]
    
    for strategy in strategies:
        mapping = _learn_with_keyfn(train_pairs, bg, strategy)
        if mapping is not None:
            return {
                'mapping': mapping,
                'strategy': strategy,
                'bg': bg,
            }
    
    return None


def apply_role_nb_rule(inp: Grid, rule: Dict) -> Grid:
    """Apply role-aware NB rule."""
    mapping = rule['mapping']
    strategy = rule['strategy']
    bg = rule['bg']
    h, w = grid_shape(inp)
    
    ir = build_object_ir(inp, bg)
    
    result = []
    for r in range(h):
        row = []
        for c in range(w):
            center = inp[r][c]
            
            if strategy == 'nb_plus_role':
                key = ir.get_nb_plus_role(r, c)
            elif strategy == 'nb_plus_compact':
                nb_part, role_part = ir.get_nb_plus_role(r, c)
                compact = ir.get_compact_role(r, c)
                key = (nb_part, compact)
            elif strategy == 'compact_only':
                key = ir.get_compact_role(r, c)
            elif strategy == 'nb_canonical_plus_compact':
                nb_part, role_part = ir.get_nb_plus_role(r, c)
                cnb = _canonical_3x3(nb_part)
                compact = ir.get_compact_role(r, c)
                key = (cnb, compact)
            elif strategy == 'signature':
                key = ir.get_role_signature(r, c)
            elif strategy == 'nb_plus_signature':
                nb_part, _ = ir.get_nb_plus_role(r, c)
                sig = ir.get_role_signature(r, c)
                key = (nb_part, sig)
            else:
                key = None
            
            if key in mapping:
                role = mapping[key]
                row.append(_resolve_role(role, center, bg))
            else:
                row.append(center)
        result.append(row)
    
    return result


def generate_role_nb_pieces(train_pairs):
    """Generate CrossPiece candidates from role-aware NB rules."""
    from arc.cross_engine import CrossPiece
    
    pieces = []
    rule = learn_role_nb_rule(train_pairs)
    if rule is not None:
        r = rule
        pieces.append(CrossPiece(
            f'role_nb:{r["strategy"]}',
            lambda inp, _r=r: apply_role_nb_rule(inp, _r),
            version=1
        ))
    return pieces
