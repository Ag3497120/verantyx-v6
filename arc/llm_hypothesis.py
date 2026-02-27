"""
arc/llm_hypothesis.py — LLM Hypothesis Generator for ARC-AGI-2

Uses Qwen2.5-7B (via Ollama) to generate transformation hypotheses
for tasks that the symbolic engine cannot solve (ver=0).

The LLM does NOT generate grids — it generates structured hypotheses
that Verantyx translates into DSL programs and verifies.
"""

import json
import subprocess
import re
from typing import List, Tuple, Optional, Dict, Any
from arc.grid import Grid, grid_shape, grid_colors, most_common_color


# --- Grid Serialization ---

def grid_to_text(grid: Grid, compact: bool = True) -> str:
    """Convert grid to compact text representation."""
    if compact:
        return '\n'.join(''.join(str(c) for c in row) for row in grid)
    return str(grid)


def task_to_prompt(train_pairs: List[Tuple[Grid, Grid]], 
                   test_input: Grid = None) -> str:
    """Format ARC task as a prompt for the LLM."""
    parts = []
    for i, (inp, out) in enumerate(train_pairs):
        h_in, w_in = grid_shape(inp)
        h_out, w_out = grid_shape(out)
        parts.append(f"Example {i+1}:")
        parts.append(f"Input ({h_in}x{w_in}):")
        parts.append(grid_to_text(inp))
        parts.append(f"Output ({h_out}x{w_out}):")
        parts.append(grid_to_text(out))
        parts.append("")
    
    if test_input:
        h, w = grid_shape(test_input)
        parts.append(f"Test Input ({h}x{w}):")
        parts.append(grid_to_text(test_input))
    
    return '\n'.join(parts)


# --- Hypothesis Categories ---

HYPOTHESIS_CATEGORIES = [
    "object_extraction",      # Extract specific object(s) from grid
    "object_transform",       # Transform objects (move, resize, recolor)
    "object_relation",        # Transform based on object relationships
    "color_mapping",          # Map colors globally or conditionally
    "pattern_fill",           # Fill regions with patterns
    "symmetry",               # Mirror, rotate, reflect
    "tiling",                 # Tile/repeat patterns
    "counting",               # Count-based transformation
    "sorting",                # Sort objects by property
    "conditional",            # If-then rules per cell/object
    "composition",            # Multi-step transformation
    "geometric",              # Lines, rays, diagonals
    "size_change",            # Upscale, downscale, crop
    "gravity",                # Move objects in direction
    "flood_fill",             # Region filling
    "dedup",                  # Remove duplicate rows/cols
    "overlay",                # Overlay/merge grids
]


SYSTEM_PROMPT = """You are an ARC-AGI-2 puzzle analyst. Given input-output grid examples, identify the transformation rule.

Respond in this exact JSON format:
{
  "category": "<one of the categories>",
  "description": "<1-2 sentence description of the transformation>",
  "steps": ["step1", "step2", ...],
  "properties": {
    "size_change": "same|grow|shrink|variable",
    "uses_color_map": true/false,
    "object_based": true/false,
    "position_dependent": true/false
  }
}

Categories: object_extraction, object_transform, object_relation, color_mapping, pattern_fill, symmetry, tiling, counting, sorting, conditional, composition, geometric, size_change, gravity, flood_fill, dedup, overlay

Colors: 0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=gray, 6=magenta, 7=orange, 8=cyan, 9=maroon

Be specific about WHAT objects are identified and HOW they are transformed. Focus on the rule, not individual examples."""


def query_ollama(prompt: str, system: str = SYSTEM_PROMPT, 
                 model: str = "qwen2.5:7b-instruct",
                 timeout: int = 30) -> Optional[str]:
    """Query Ollama and return response text."""
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 512,
            }
        }
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", "http://localhost:11434/api/chat",
             "-d", json.dumps(payload)],
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            return None
        resp = json.loads(result.stdout)
        return resp.get("message", {}).get("content", "")
    except Exception as e:
        return None


def parse_hypothesis(response: str) -> Optional[Dict[str, Any]]:
    """Parse LLM response into structured hypothesis."""
    if not response:
        return None
    
    # Try to find the outermost JSON object with "category"
    # Handle nested JSON strings by finding balanced braces
    def find_json_objects(text):
        results = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                depth = 0
                start = i
                for j in range(i, len(text)):
                    if text[j] == '{': depth += 1
                    elif text[j] == '}': depth -= 1
                    if depth == 0:
                        candidate = text[start:j+1]
                        if '"category"' in candidate:
                            results.append(candidate)
                        break
            i += 1
        return results
    
    # Try code block first
    code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Find all JSON objects
    candidates = find_json_objects(response)
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict) and 'category' in obj:
                # If description is itself a JSON string, parse it
                if isinstance(obj.get('description'), str) and obj['description'].startswith('{'):
                    try:
                        inner = json.loads(obj['description'])
                        if isinstance(inner, dict) and 'category' in inner:
                            return inner
                    except:
                        pass
                return obj
        except json.JSONDecodeError:
            continue
    
    # Fallback: extract category from text
    for cat in HYPOTHESIS_CATEGORIES:
        if cat in response.lower():
            return {"category": cat, "description": response[:200], "steps": [], "properties": {}}
    
    return None


def generate_hypothesis(train_pairs: List[Tuple[Grid, Grid]], 
                        test_input: Grid = None,
                        model: str = "qwen2.5:7b-instruct") -> Optional[Dict[str, Any]]:
    """Generate a transformation hypothesis for an ARC task."""
    prompt = task_to_prompt(train_pairs, test_input)
    response = query_ollama(prompt, model=model)
    return parse_hypothesis(response)


# --- Hypothesis to DSL Translation ---

def hypothesis_to_pieces(hypothesis: Dict[str, Any], 
                         train_pairs: List[Tuple[Grid, Grid]]) -> List:
    """
    Translate an LLM hypothesis into candidate CrossPieces.
    
    This is where the LLM's semantic understanding meets
    Verantyx's symbolic execution.
    """
    from arc.cross_engine import CrossPiece, CrossSimulator, _generate_cross_pieces
    
    if not hypothesis:
        return []
    
    category = hypothesis.get("category", "")
    description = hypothesis.get("description", "").lower()
    steps = hypothesis.get("steps", [])
    props = hypothesis.get("properties", {})
    
    pieces = []
    
    # Generate targeted pieces based on hypothesis category
    if category == "object_extraction":
        pieces.extend(_try_extraction_pieces(train_pairs, description))
    
    elif category == "object_transform":
        pieces.extend(_try_object_transform_pieces(train_pairs, description))
    
    elif category == "object_relation":
        pieces.extend(_try_relation_pieces(train_pairs, description))
    
    elif category == "color_mapping":
        pieces.extend(_try_color_mapping_pieces(train_pairs, description))
    
    elif category == "pattern_fill":
        pieces.extend(_try_pattern_fill_pieces(train_pairs, description))
    
    elif category == "symmetry":
        pieces.extend(_try_symmetry_pieces(train_pairs, description))
    
    elif category == "tiling":
        pieces.extend(_try_tiling_pieces(train_pairs, description))
    
    elif category == "counting":
        pieces.extend(_try_counting_pieces(train_pairs, description))
    
    elif category == "sorting":
        pieces.extend(_try_sorting_pieces(train_pairs, description))
    
    elif category == "geometric":
        pieces.extend(_try_geometric_pieces(train_pairs, description))
    
    elif category == "gravity":
        pieces.extend(_try_gravity_pieces(train_pairs, description))
    
    elif category == "composition":
        pieces.extend(_try_composition_pieces(train_pairs, description, steps))
    
    elif category == "size_change":
        pieces.extend(_try_size_change_pieces(train_pairs, description))
    
    # Always also try generic pieces as fallback
    pieces.extend(_generate_cross_pieces(train_pairs))
    
    return pieces


def _try_extraction_pieces(train_pairs, desc):
    """Generate extraction-focused pieces."""
    from arc.cross_engine import CrossPiece
    pieces = []
    
    # Try various extraction strategies
    try:
        from arc.objects import detect_objects
        for inp, out in train_pairs[:1]:
            objs = detect_objects(inp)
            h_out, w_out = grid_shape(out)
            # Find object matching output size
            for obj in objs:
                if obj.get('h') == h_out and obj.get('w') == w_out:
                    # This object likely needs extraction
                    color = obj.get('color', -1)
                    def _extract(grid, c=color):
                        from arc.objects import detect_objects
                        for o in detect_objects(grid):
                            if o.get('color') == c:
                                r, c_pos = o['top'], o['left']
                                return [row[c_pos:c_pos+o['w']] for row in grid[r:r+o['h']]]
                        return None
                    pieces.append(CrossPiece(f'llm:extract_color_{color}', _extract))
    except Exception:
        pass
    
    # Extract by size (largest, smallest, unique)
    try:
        from arc.cross_solver import _generate_whole_grid_candidates
        wg = _generate_whole_grid_candidates(train_pairs)
        for w in wg:
            pieces.append(CrossPiece(f'llm:wg:{w.name}', w.apply))
    except Exception:
        pass
    
    return pieces


def _try_object_transform_pieces(train_pairs, desc):
    """Generate object-transform pieces based on description."""
    from arc.cross_engine import CrossPiece
    pieces = []
    
    # Check for keywords in description
    keywords_to_fn = {
        'move': 'translate',
        'shift': 'translate', 
        'rotate': 'rotate',
        'flip': 'flip',
        'mirror': 'mirror',
        'scale': 'scale',
        'grow': 'grow',
        'shrink': 'shrink',
        'recolor': 'recolor',
        'replace': 'replace',
    }
    
    matched = False
    for kw, fn_type in keywords_to_fn.items():
        if kw in desc:
            pieces.extend(_generate_transform_by_type(train_pairs, fn_type))
            matched = True
    
    if not matched:
        # Always generate base pieces for object_transform
        pieces.extend(_generate_transform_by_type(train_pairs, 'generic'))
    
    return pieces


def _try_relation_pieces(train_pairs, desc):
    """Generate relation-based pieces."""
    pieces = []
    # Object relationships: adjacency, containment, alignment
    try:
        from arc.conditional_transform import learn_conditional_object_transform
        params = learn_conditional_object_transform(train_pairs)
        if params:
            from arc.conditional_transform import apply_conditional_object_transform
            from arc.cross_engine import CrossPiece
            def _apply(inp, p=params):
                return apply_conditional_object_transform(inp, p)
            pieces.append(CrossPiece('llm:cond_obj_transform', _apply))
    except Exception:
        pass
    return pieces


def _try_color_mapping_pieces(train_pairs, desc):
    """Generate color-mapping pieces."""
    from arc.cross_engine import CrossPiece
    pieces = []
    
    # Learn global color map from examples
    try:
        inp0, out0 = train_pairs[0]
        h, w = grid_shape(inp0)
        h2, w2 = grid_shape(out0)
        if h == h2 and w == w2:
            # Same size — try cell-wise color mapping
            color_map = {}
            for r in range(h):
                for c in range(w):
                    ci = inp0[r][c]
                    co = out0[r][c]
                    if ci in color_map:
                        if color_map[ci] != co:
                            color_map = None
                            break
                    else:
                        color_map[ci] = co
                if color_map is None:
                    break
            
            if color_map:
                def _apply_cmap(inp, cmap=dict(color_map)):
                    return [[cmap.get(c, c) for c in row] for row in inp]
                pieces.append(CrossPiece('llm:global_color_map', _apply_cmap))
    except Exception:
        pass
    
    return pieces


def _try_pattern_fill_pieces(train_pairs, desc):
    """Generate pattern-fill pieces."""
    pieces = []
    try:
        from arc.flood_fill import learn_flood_fill_region, apply_flood_fill_region
        params = learn_flood_fill_region(train_pairs)
        if params:
            from arc.cross_engine import CrossPiece
            def _apply(inp, p=params):
                return apply_flood_fill_region(inp, p)
            pieces.append(CrossPiece('llm:flood_fill', _apply))
    except Exception:
        pass
    return pieces


def _try_symmetry_pieces(train_pairs, desc):
    """Generate symmetry-based pieces."""
    from arc.cross_engine import CrossPiece
    pieces = []
    
    fns = {
        'flip_h': lambda g: [row[::-1] for row in g],
        'flip_v': lambda g: g[::-1],
        'rot90': lambda g: [list(row) for row in zip(*g[::-1])],
        'rot180': lambda g: [row[::-1] for row in g[::-1]],
        'rot270': lambda g: [list(row) for row in zip(*g)][::-1],
        'transpose': lambda g: [list(row) for row in zip(*g)],
    }
    
    for name, fn in fns.items():
        pieces.append(CrossPiece(f'llm:sym:{name}', fn))
    
    # Symmetry fill
    try:
        from arc.cross_solver import _generate_whole_grid_candidates
        for wg in _generate_whole_grid_candidates(train_pairs):
            if 'sym' in wg.name.lower():
                pieces.append(CrossPiece(f'llm:wg:{wg.name}', wg.apply))
    except Exception:
        pass
    
    return pieces


def _try_tiling_pieces(train_pairs, desc):
    """Generate tiling pieces."""
    from arc.cross_engine import CrossPiece
    pieces = []
    
    inp0, out0 = train_pairs[0]
    h_in, w_in = grid_shape(inp0)
    h_out, w_out = grid_shape(out0)
    
    # Check if output is a multiple of input
    if h_out > 0 and w_out > 0 and h_in > 0 and w_in > 0:
        ry = h_out // h_in if h_in > 0 else 0
        rx = w_out // w_in if w_in > 0 else 0
        if ry > 0 and rx > 0 and h_out == h_in * ry and w_out == w_in * rx:
            def _tile(inp, _ry=ry, _rx=rx):
                return [row * _rx for row in inp] * _ry
            pieces.append(CrossPiece(f'llm:tile_{ry}x{rx}', _tile))
    
    return pieces


def _try_counting_pieces(train_pairs, desc):
    """Generate counting-based pieces."""
    from arc.cross_engine import CrossPiece
    pieces = []
    
    # Count objects → 1x1 output
    inp0, out0 = train_pairs[0]
    h_out, w_out = grid_shape(out0)
    
    if h_out == 1 and w_out == 1:
        # Output is a single cell — likely counting
        try:
            from arc.objects import detect_objects
            objs = detect_objects(inp0)
            n = len(objs)
            expected = out0[0][0]
            if n == expected:
                def _count(inp):
                    from arc.objects import detect_objects
                    return [[len(detect_objects(inp))]]
                pieces.append(CrossPiece('llm:count_objects', _count))
        except Exception:
            pass
    
    return pieces


def _try_sorting_pieces(train_pairs, desc):
    """Generate sorting-based pieces."""
    return []  # TODO: implement sorting heuristics


def _try_geometric_pieces(train_pairs, desc):
    """Generate geometric (line/ray) pieces."""
    pieces = []
    try:
        from arc.line_ray_primitives import learn_line_ray_from_objects, apply_line_ray_from_objects
        params = learn_line_ray_from_objects(train_pairs)
        if params:
            from arc.cross_engine import CrossPiece
            def _apply(inp, p=params):
                return apply_line_ray_from_objects(inp, p)
            pieces.append(CrossPiece('llm:line_ray', _apply))
    except Exception:
        pass
    return pieces


def _try_gravity_pieces(train_pairs, desc):
    """Generate gravity pieces."""
    from arc.cross_engine import CrossPiece
    pieces = []
    
    directions = ['down', 'up', 'left', 'right']
    for d in directions:
        if d in desc:
            def _gravity(inp, direction=d):
                from arc.grid import grid_shape
                h, w = grid_shape(inp)
                bg = most_common_color(inp)
                result = [[bg]*w for _ in range(h)]
                
                if direction == 'down':
                    for c in range(w):
                        non_bg = [inp[r][c] for r in range(h) if inp[r][c] != bg]
                        for i, v in enumerate(non_bg):
                            result[h - len(non_bg) + i][c] = v
                elif direction == 'up':
                    for c in range(w):
                        non_bg = [inp[r][c] for r in range(h) if inp[r][c] != bg]
                        for i, v in enumerate(non_bg):
                            result[i][c] = v
                elif direction == 'left':
                    for r in range(h):
                        non_bg = [inp[r][c] for c in range(w) if inp[r][c] != bg]
                        for i, v in enumerate(non_bg):
                            result[r][i] = v
                elif direction == 'right':
                    for r in range(h):
                        non_bg = [inp[r][c] for c in range(w) if inp[r][c] != bg]
                        for i, v in enumerate(non_bg):
                            result[r][w - len(non_bg) + i] = v
                return result
            pieces.append(CrossPiece(f'llm:gravity_{d}', _gravity))
    
    return pieces


def _try_composition_pieces(train_pairs, desc, steps):
    """Generate multi-step composition pieces."""
    from arc.cross_engine import CrossPiece, _generate_cross_pieces
    pieces = []
    
    # Use existing cross pieces and try 2-step compositions
    base_pieces = _generate_cross_pieces(train_pairs)
    
    # Limit combinatorial explosion
    if len(base_pieces) > 20:
        base_pieces = base_pieces[:20]
    
    for p1 in base_pieces[:10]:
        for p2 in base_pieces[:10]:
            if p1.name == p2.name:
                continue
            def _compose(inp, _p1=p1, _p2=p2):
                mid = _p1.apply(inp)
                if mid is None:
                    return None
                return _p2.apply(mid)
            pieces.append(CrossPiece(f'llm:compose:{p1.name}+{p2.name}', _compose))
    
    return pieces


def _try_size_change_pieces(train_pairs, desc):
    """Generate size-change pieces."""
    from arc.cross_engine import CrossPiece
    pieces = []
    
    inp0, out0 = train_pairs[0]
    h_in, w_in = grid_shape(inp0)
    h_out, w_out = grid_shape(out0)
    
    # Upscale
    for factor in [2, 3, 4, 5]:
        if h_out == h_in * factor and w_out == w_in * factor:
            def _upscale(inp, f=factor):
                return [[c for c in row for _ in range(f)] for row in inp for _ in range(f)]
            pieces.append(CrossPiece(f'llm:upscale_{factor}x', _upscale))
    
    # Downscale
    for factor in [2, 3, 4, 5]:
        if h_in == h_out * factor and w_in == w_out * factor:
            def _downscale(inp, f=factor):
                h, w = grid_shape(inp)
                return [[inp[r*f][c*f] for c in range(w//f)] for r in range(h//f)]
            pieces.append(CrossPiece(f'llm:downscale_{factor}x', _downscale))
    
    return pieces


def _generate_transform_by_type(train_pairs, fn_type):
    """Generate transform pieces by type."""
    # Delegate to existing machinery
    from arc.cross_engine import _generate_cross_pieces
    return _generate_cross_pieces(train_pairs)


# --- Main Integration Point ---

def solve_with_llm_hypothesis(train_pairs: List[Tuple[Grid, Grid]],
                               test_inputs: List[Grid],
                               model: str = "qwen2.5:7b-instruct") -> Tuple[Optional[List[List[Grid]]], Optional[List], Optional[Dict]]:
    """
    LLM-guided solving:
    1. Ask LLM for hypothesis
    2. Translate hypothesis to pieces
    3. Verify pieces against training
    4. Return predictions if verified
    """
    from arc.cross_engine import CrossSimulator
    sim = CrossSimulator()
    
    # Generate hypothesis
    hypothesis = generate_hypothesis(train_pairs, test_inputs[0] if test_inputs else None, model=model)
    
    if not hypothesis:
        return None, None, None
    
    # Translate to pieces
    pieces = hypothesis_to_pieces(hypothesis, train_pairs)
    
    # Verify
    verified = []
    for piece in pieces:
        if sim.verify(piece, train_pairs):
            verified.append(('cross', piece))
            if len(verified) >= 2:
                break
    
    # If no full verify, try refinement with best partial match
    if not verified:
        best_score = 0
        best_piece = None
        for piece in pieces[:100]:
            score = sim.partial_verify(piece, train_pairs)
            if score > best_score:
                best_score = score
                best_piece = piece
        
        if best_piece and best_score >= 0.5:
            # Try composition: best_piece + another piece
            for piece2 in pieces[:50]:
                if piece2.name == best_piece.name:
                    continue
                from arc.cross_engine import CrossPiece
                def _refine(inp, _p1=best_piece, _p2=piece2):
                    mid = _p1.apply(inp)
                    if mid is None:
                        return None
                    return _p2.apply(mid)
                refine_piece = CrossPiece(f'llm:refine:{best_piece.name}+{piece2.name}', _refine)
                if sim.verify(refine_piece, train_pairs):
                    verified.append(('cross', refine_piece))
                    break
            
            # Also try: piece2 + best_piece
            if not verified:
                for piece2 in pieces[:50]:
                    if piece2.name == best_piece.name:
                        continue
                    from arc.cross_engine import CrossPiece
                    def _refine2(inp, _p1=piece2, _p2=best_piece):
                        mid = _p1.apply(inp)
                        if mid is None:
                            return None
                        return _p2.apply(mid)
                    refine_piece2 = CrossPiece(f'llm:refine:{piece2.name}+{best_piece.name}', _refine2)
                    if sim.verify(refine_piece2, train_pairs):
                        verified.append(('cross', refine_piece2))
                        break
            
            # Try convergent application
            if not verified and best_score >= 0.8:
                def _converge(inp, _piece=best_piece):
                    x = inp
                    for _ in range(20):
                        try:
                            y = _piece.apply(x)
                        except Exception:
                            break
                        if y is None or y == x:
                            break
                        x = y
                    return x
                conv_piece = CrossPiece(f'llm:converge:{best_piece.name}', _converge)
                if sim.verify(conv_piece, train_pairs):
                    verified.append(('cross', conv_piece))
    
    if not verified:
        return None, None, hypothesis
    
    # Apply to test
    predictions = []
    for test_inp in test_inputs:
        preds = []
        for _, piece in verified:
            try:
                result = piece.apply(test_inp)
                if result is not None:
                    preds.append(result)
            except Exception:
                pass
        predictions.append(preds if preds else [])
    
    return predictions, verified, hypothesis
