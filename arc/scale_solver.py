"""
arc/scale_solver.py — Scale/upscale solver

Handles tasks where output = input scaled by integer factor,
possibly with modifications per cell.
"""

import numpy as np
from typing import List, Tuple
from collections import Counter
from arc.cross_engine import CrossPiece

Grid = List[List[int]]

def _g(g): return np.array(g, dtype=int)
def _l(a): return a.tolist()


def learn_simple_upscale(train_pairs):
    """Output = each cell of input expanded to NxN block"""
    for factor in range(2, 6):
        all_match = True
        for inp_g, out_g in train_pairs:
            inp, out = _g(inp_g), _g(out_g)
            ih, iw = inp.shape
            oh, ow = out.shape
            
            if oh != ih * factor or ow != iw * factor:
                all_match = False; break
            
            for r in range(ih):
                for c in range(iw):
                    block = out[r*factor:(r+1)*factor, c*factor:(c+1)*factor]
                    if not np.all(block == inp[r, c]):
                        all_match = False; break
                if not all_match: break
        
        if all_match:
            return {'factor': factor, 'type': 'simple'}
    return None


def learn_upscale_with_pattern(train_pairs):
    """Output = each cell mapped to an NxN pattern block based on cell value"""
    for factor in range(2, 6):
        color_to_block = {}
        all_match = True
        
        for inp_g, out_g in train_pairs:
            inp, out = _g(inp_g), _g(out_g)
            ih, iw = inp.shape
            oh, ow = out.shape
            
            if oh != ih * factor or ow != iw * factor:
                all_match = False; break
            
            for r in range(ih):
                for c in range(iw):
                    color = int(inp[r, c])
                    block = out[r*factor:(r+1)*factor, c*factor:(c+1)*factor]
                    block_key = block.tobytes()
                    
                    if color in color_to_block:
                        if color_to_block[color].tobytes() != block_key:
                            all_match = False; break
                    else:
                        color_to_block[color] = block.copy()
                if not all_match: break
        
        if all_match and color_to_block:
            return {'factor': factor, 'type': 'pattern', 
                    'blocks': {k: v.tolist() for k, v in color_to_block.items()}}
    return None


def learn_upscale_with_border(train_pairs):
    """Output = each cell expanded to NxN with border of different color"""
    for factor in range(2, 6):
        border_color = None
        all_match = True
        
        for inp_g, out_g in train_pairs:
            inp, out = _g(inp_g), _g(out_g)
            ih, iw = inp.shape
            oh, ow = out.shape
            
            if oh != ih * factor or ow != iw * factor:
                all_match = False; break
            
            for r in range(ih):
                for c in range(iw):
                    block = out[r*factor:(r+1)*factor, c*factor:(c+1)*factor]
                    # Inner should be inp[r,c], border should be consistent
                    inner = block[1:-1, 1:-1] if factor >= 3 else None
                    
                    if factor >= 3:
                        if not np.all(inner == inp[r, c]):
                            all_match = False; break
                        # Check border
                        border_vals = set()
                        for i in range(factor):
                            border_vals.add(int(block[0, i]))
                            border_vals.add(int(block[factor-1, i]))
                            border_vals.add(int(block[i, 0]))
                            border_vals.add(int(block[i, factor-1]))
                        border_vals -= {int(inp[r, c])}
                        
                        if len(border_vals) == 1:
                            bc = border_vals.pop()
                            if border_color is None:
                                border_color = bc
                            elif border_color != bc:
                                all_match = False; break
                        elif len(border_vals) == 0:
                            pass  # same as inner
                        else:
                            all_match = False; break
                if not all_match: break
        
        if all_match and border_color is not None:
            return {'factor': factor, 'type': 'border', 'border_color': border_color}
    return None


def _apply(inp_g, rule):
    inp = _g(inp_g)
    ih, iw = inp.shape
    factor = rule['factor']
    
    if rule['type'] == 'simple':
        out = np.zeros((ih * factor, iw * factor), dtype=int)
        for r in range(ih):
            for c in range(iw):
                out[r*factor:(r+1)*factor, c*factor:(c+1)*factor] = inp[r, c]
        return _l(out)
    
    elif rule['type'] == 'pattern':
        out = np.zeros((ih * factor, iw * factor), dtype=int)
        blocks = {k: np.array(v) for k, v in rule['blocks'].items()}
        for r in range(ih):
            for c in range(iw):
                color = int(inp[r, c])
                if color in blocks:
                    out[r*factor:(r+1)*factor, c*factor:(c+1)*factor] = blocks[color]
                else:
                    out[r*factor:(r+1)*factor, c*factor:(c+1)*factor] = color
        return _l(out)
    
    elif rule['type'] == 'border':
        out = np.full((ih * factor, iw * factor), rule['border_color'], dtype=int)
        for r in range(ih):
            for c in range(iw):
                out[r*factor+1:(r+1)*factor-1, c*factor+1:(c+1)*factor-1] = inp[r, c]
        return _l(out)
    
    return None


def _verify(fn, train_pairs):
    for inp, out in train_pairs:
        pred = fn(inp)
        if pred is None or not np.array_equal(_g(pred), _g(out)):
            return False
    return True


def generate_scale_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    pieces = []
    if not train_pairs:
        return pieces
    
    inp0, out0 = _g(train_pairs[0][0]), _g(train_pairs[0][1])
    if out0.size <= inp0.size:
        return pieces
    
    for learn_fn in [learn_simple_upscale, learn_upscale_with_pattern, learn_upscale_with_border]:
        try:
            rule = learn_fn(train_pairs)
            if rule is None: continue
            fn = lambda inp, r=rule: _apply(inp, r)
            if _verify(fn, train_pairs):
                pieces.append(CrossPiece(name=f"scale:{rule['type']}_{rule['factor']}x", apply_fn=fn))
                return pieces
        except Exception:
            continue
    
    return pieces
