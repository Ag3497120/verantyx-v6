"""
arc/llm_router.py — LLM as Router for Verantyx Cross Engine

The LLM does NOT solve puzzles. It classifies the transformation type
and suggests which Verantyx primitives/phases to prioritize.

This narrows the search space so the symbolic engine can find solutions faster.
"""

import json
import os
import urllib.request
from typing import List, Tuple, Optional, Dict
from arc.grid import Grid, grid_shape, grid_colors, most_common_color


DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-1c9551e705dd4fbfbdcab991cc924526")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


# Map of Verantyx engine phases/primitives
VERANTYX_PRIMITIVES = {
    "neighborhood_rule": "Cell-level neighborhood rules (NB). Each cell's output depends on its local neighbors.",
    "color_map": "Global or conditional color remapping. Every cell of color A becomes color B.",
    "extract_object": "Extract a specific object (largest, smallest, unique color, unique shape).",
    "object_recolor": "Detect objects and recolor them based on size, shape, position, or count.",
    "symmetry": "Mirror, rotate, transpose, or complete a symmetric pattern.",
    "tiling": "Tile/repeat the input or a sub-pattern to fill the output.",
    "gravity": "Move non-background cells in a direction (up/down/left/right) until they hit something.",
    "flood_fill": "Fill enclosed regions or connected components with a color.",
    "line_draw": "Draw lines, rays, or diagonals connecting objects or extending from points.",
    "crop_extract": "Crop a subgrid, extract a bounding box, or select a region.",
    "upscale_downscale": "Scale the grid by an integer factor (2x, 3x, etc).",
    "dedup": "Remove duplicate rows/columns or repeated patterns.",
    "overlay_merge": "Overlay or merge multiple grids/layers (AND, OR, XOR operations).",
    "separator_split": "Split grid by separator lines, apply operation to panels.",
    "pattern_stamp": "Stamp a pattern at object locations or based on object properties.",
    "composition_2step": "Two-step composition: apply primitive A, then primitive B.",
    "conditional_transform": "Different transforms for different objects based on properties.",
    "counting_output": "Output depends on counting objects, colors, or sizes.",
    "sort_arrange": "Sort or rearrange objects by size, color, or position.",
    "iterative_converge": "Apply a rule repeatedly until the grid converges (stops changing).",
}


ROUTER_SYSTEM = """You are a classifier for ARC-AGI-2 puzzle transformations. Your job is to identify WHICH type of transformation is happening, so the symbolic solver knows which primitives to try.

Given input-output examples, respond with EXACTLY this JSON format:
{
  "primary": "<primitive_name>",
  "secondary": "<primitive_name or null>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<1 sentence explaining why>"
}

Available primitives:
- neighborhood_rule: Cell output depends on local neighbors
- color_map: Global color remapping (A→B for all cells)
- extract_object: Extract specific object(s) from grid
- object_recolor: Recolor objects by size/shape/position
- symmetry: Mirror, rotate, transpose, complete symmetry
- tiling: Tile/repeat pattern to fill output
- gravity: Move cells in a direction until blocked
- flood_fill: Fill enclosed regions
- line_draw: Draw lines/rays connecting or extending from objects
- crop_extract: Crop subgrid or extract bounding box
- upscale_downscale: Scale grid by integer factor
- dedup: Remove duplicate rows/columns
- overlay_merge: Overlay/merge grids (AND/OR/XOR)
- separator_split: Split by separators, operate on panels
- pattern_stamp: Stamp pattern at object locations
- composition_2step: Two primitives applied in sequence
- conditional_transform: Different rules for different objects
- counting_output: Output based on counting
- sort_arrange: Sort/rearrange objects
- iterative_converge: Apply rule until stable

Pick the MOST SPECIFIC primitive that matches. If it's two steps, use "composition_2step" as primary and the two steps as secondary (e.g. "extract_object+color_map").

Colors: 0=black 1=blue 2=red 3=green 4=yellow 5=gray 6=magenta 7=orange 8=cyan 9=maroon"""


def grid_to_text(grid: Grid) -> str:
    return '\n'.join(''.join(str(c) for c in row) for row in grid)


def call_deepseek(messages: list, model: str = "deepseek-chat",
                  temperature: float = 0.0, max_tokens: int = 256,
                  timeout: int = 30) -> Optional[str]:
    """Call DeepSeek API."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    
    req = urllib.request.Request(
        DEEPSEEK_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
    )
    
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except Exception:
        return None


def classify_task(train_pairs: List[Tuple[Grid, Grid]],
                  model: str = "deepseek-chat") -> Optional[Dict]:
    """
    Classify an ARC task into Verantyx primitive categories.
    Returns routing info for the symbolic engine.
    """
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
    
    prompt = '\n'.join(parts)
    
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    
    response = call_deepseek(messages, model=model)
    if not response:
        return None
    
    # Parse JSON
    import re
    # Try direct parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try extracting JSON from text
    m = re.search(r'\{.*?\}', response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


def route_to_phases(classification: Dict) -> List[str]:
    """
    Convert LLM classification to ordered list of Verantyx phases to try.
    The engine will try these phases first before falling back to brute force.
    """
    if not classification:
        return []
    
    primary = classification.get("primary", "")
    secondary = classification.get("secondary", "")
    
    phases = []
    
    # Map primitives to engine phases
    PHASE_MAP = {
        "neighborhood_rule": ["phase1_nb", "phase1_cross_nb"],
        "color_map": ["phase7_puzzle", "phase1_standalone"],
        "extract_object": ["phase1.5_standalone", "phase2_cross_pieces"],
        "object_recolor": ["phase2_cross_pieces", "phase7_puzzle"],
        "symmetry": ["phase1.5_standalone", "phase7_puzzle"],
        "tiling": ["phase1.55_cross_universe", "phase1.5_standalone"],
        "gravity": ["phase1.5_standalone", "phase2_cross_pieces"],
        "flood_fill": ["phase2_cross_pieces", "phase7_puzzle"],
        "line_draw": ["phase2_cross_pieces", "phase7_puzzle"],
        "crop_extract": ["phase1.5_standalone", "phase9_cegis"],
        "upscale_downscale": ["phase7_puzzle", "phase1.5_standalone"],
        "dedup": ["phase9_cegis", "phase1.5_standalone"],
        "overlay_merge": ["phase7_puzzle", "phase1.56_cross3d"],
        "separator_split": ["phase7_puzzle", "phase1.56_cross3d"],
        "pattern_stamp": ["phase2_cross_pieces", "phase1.6_convergent"],
        "composition_2step": ["phase3_composition", "phase8_ptree"],
        "conditional_transform": ["phase2_cross_pieces", "phase8_ptree"],
        "counting_output": ["phase7_puzzle", "phase2_cross_pieces"],
        "sort_arrange": ["phase8_ptree", "phase2_cross_pieces"],
        "iterative_converge": ["phase1.6_convergent", "phase4_iterative"],
    }
    
    if primary in PHASE_MAP:
        phases.extend(PHASE_MAP[primary])
    
    if secondary and secondary in PHASE_MAP:
        for p in PHASE_MAP[secondary]:
            if p not in phases:
                phases.append(p)
    
    # Handle composition notation like "extract_object+color_map"
    if secondary and '+' in str(secondary):
        parts = str(secondary).split('+')
        for part in parts:
            part = part.strip()
            if part in PHASE_MAP:
                for p in PHASE_MAP[part]:
                    if p not in phases:
                        phases.append(p)
    
    return phases


def batch_classify(task_ids: List[str], data_dir: str,
                   model: str = "deepseek-chat",
                   output_file: str = "llm_classifications.json") -> Dict:
    """
    Batch classify multiple tasks and cache results.
    """
    # Load existing cache
    cache = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            cache = json.load(f)
    
    for i, tid in enumerate(task_ids):
        if tid in cache:
            continue
        
        task_path = os.path.join(data_dir, f"{tid}.json")
        if not os.path.exists(task_path):
            continue
        
        with open(task_path) as f:
            data = json.load(f)
        
        train_pairs = [(ex['input'], ex['output']) for ex in data['train']]
        
        classification = classify_task(train_pairs, model=model)
        if classification:
            cache[tid] = classification
        
        # Save periodically
        if (i + 1) % 10 == 0:
            with open(output_file, 'w') as f:
                json.dump(cache, f, indent=2)
            print(f"  Classified {i+1} tasks, cached {len(cache)}")
    
    # Final save
    with open(output_file, 'w') as f:
        json.dump(cache, f, indent=2)
    
    return cache
