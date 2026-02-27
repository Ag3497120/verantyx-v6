"""Debug: show full hypotheses and what pieces are generated."""
import os, json, re, time, sys
from arc.llm_hypothesis import generate_hypothesis, hypothesis_to_pieces, task_to_prompt
from arc.cross_engine import CrossSimulator
from arc.grid import grid_eq

data_dir = "/tmp/arc-agi-2/data/training"

# Get ver=0 IDs
ver0 = []
with open("arc_v61_full.log") as f:
    for l in f:
        m = re.match(r'\s*\[(\d+)/1000\]\s+✗\s+[\d.]+s/t\s+(\w+)\s+ver=0', l)
        if m:
            ver0.append(m.group(2))

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 3

for tid in ver0[:limit]:
    path = os.path.join(data_dir, f"{tid}.json")
    with open(path) as f:
        data = json.load(f)
    
    train = [(ex['input'], ex['output']) for ex in data['train']]
    test_in = [ex['input'] for ex in data['test']]
    test_out = [ex.get('output') for ex in data['test']]
    
    print(f"=== {tid} ===")
    
    # Show task shape
    for i, (inp, out) in enumerate(train):
        print(f"  train[{i}]: {len(inp)}x{len(inp[0])} → {len(out)}x{len(out[0])}")
    
    # Generate hypothesis
    hyp = generate_hypothesis(train, test_in[0])
    print(f"  Hypothesis: {json.dumps(hyp, indent=2)[:500]}")
    
    # Generate pieces
    pieces = hypothesis_to_pieces(hyp, train)
    print(f"  Pieces generated: {len(pieces)}")
    
    # Check partial matches
    sim = CrossSimulator()
    best_score = 0
    best_piece = None
    for p in pieces[:50]:
        score = sim.partial_verify(p, train)
        if score > best_score:
            best_score = score
            best_piece = p
    
    print(f"  Best partial match: {best_score:.2f} ({best_piece.name if best_piece else 'none'})")
    
    # Did any verify?
    verified = [p for p in pieces if sim.verify(p, train)]
    print(f"  Fully verified: {len(verified)}")
    if verified:
        r = verified[0].apply(test_in[0])
        if r and test_out[0] and grid_eq(r, test_out[0]):
            print(f"  ✓ CORRECT!")
    print()
