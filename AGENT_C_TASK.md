# Agent C Task: GGUF Fix + Knowledge Extraction from 13 Shards

## Context
You are working on **Verantyx V6**, a rule-based reasoning system for HLE (Humanity's Last Exam).
- Workspace: `/Users/motonishikoudai/.openclaw/workspace/verantyx_v6/`
- GGUF files location: check `~/` or `/Volumes/` for large GGUF files
- Goal: Extract knowledge from DeepSeek V3-0324 GGUF weights to improve HLE score
- Deadline: Feb 28, 2026 (9 days)

## GGUF Status
- Shards 00001~00010, 00012, 00014, 00015: ✅ Downloaded
- Shard 00011: wget returned 404 Not Found
- Shard 00013: wget returned 404 Not Found
- wget logs: `/tmp/wget_shard00011.log`, `/tmp/wget_shard00013.log`

## Your Mission

### Part 1: Fix GGUF Download (Shards 00011 and 00013)

#### Step 1a: Find correct URLs
The 404 might be because:
- The file was renamed/reorganized on HuggingFace
- The URL pattern changed
- Files are now served via XET (different protocol)

Check the HuggingFace page for the correct current URLs:
```bash
# Try HF Hub API to list files
curl -s "https://huggingface.co/api/models/unsloth/DeepSeek-V3-0324-GGUF" | python3 -c "
import json, sys
data = json.load(sys.stdin)
siblings = data.get('siblings', [])
for s in siblings:
    if '00011' in s['rfilename'] or '00013' in s['rfilename']:
        print(s['rfilename'])
" 2>/dev/null || echo "Direct API call failed, trying alternative..."

# Alternative: check huggingface-cli
python3 -c "
from huggingface_hub import HfFileSystem
fs = HfFileSystem()
files = fs.ls('unsloth/DeepSeek-V3-0324-GGUF', detail=False)
for f in files:
    if '00011' in f or '00013' in f:
        print(f)
" 2>/dev/null
```

#### Step 1b: Start correct downloads
If URLs are found, start wget with correct URLs:
```bash
# Example (replace with actual correct filename)
wget -c --retry-connrefused --timeout=30 \
  "https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF/resolve/main/CORRECT_FILENAME_00011.gguf" \
  -O ~/DeepSeek-V3-0324-Q8_0-00011-of-00015.gguf \
  --progress=dot:mega \
  2>&1 > /tmp/wget_shard00011_fixed.log &
echo "PID: $!"
```

### Part 2: Knowledge Extraction from Existing 13 Shards

This is the more impactful part. We have 13 of 15 shards already downloaded.

#### Step 2a: Find where GGUF files are
```bash
find ~ -name "*.gguf" -size +1G 2>/dev/null | head -20
find /Volumes -name "*.gguf" -size +1G 2>/dev/null | head -20 || true
```

#### Step 2b: Check the expert_loader.py
The GGUF reader is implemented in `knowledge/expert_loader.py`.
Check if it works with existing shards:
```bash
cd /Users/motonishikoudai/.openclaw/workspace/verantyx_v6
python3 -c "
import sys
sys.path.insert(0, '.')
from knowledge.expert_loader import ExpertLoader
# Try loading from first available shard
loader = ExpertLoader()
print('ExpertLoader OK')
print(loader.__doc__ or 'no doc')
" 2>&1 | head -30
```

#### Step 2c: Implement mine_trace_from_shard.py
This script doesn't exist yet. Create it to:
1. Load existing GGUF shards (13 shards available)
2. Extract knowledge traces — for HLE question types, find which expert neurons activate
3. Map activations to knowledge domains
4. Add high-confidence knowledge to `pieces/piece_db.jsonl`

The approach:
```python
# mine_trace_from_shard.py
"""
Mine knowledge from DeepSeek V3-0324 GGUF shards.
Strategy: For each piece in piece_db, check which experts strongly activate
on the question text. Use this to improve piece selection.
"""

import sys, json, numpy as np
sys.path.insert(0, '.')
from knowledge.expert_loader import ExpertLoader

# Load piece db
pieces = []
with open('pieces/piece_db.jsonl') as f:
    for line in f:
        pieces.append(json.loads(line))

print(f"Loaded {len(pieces)} pieces")

# For each piece, compute its "signature" in expert space
# ... (implement based on ExpertLoader capabilities)
```

#### Step 2d: Alternative — Use concept_dirs.npy for better matching
Thunder Compute SVD assets are at:
`~/avh_math/avh_math/db/moe_sparse_cross_600b_real/`

- `concept_dirs.npy`: (15104, 4, 7168) — SVD directions for all experts
- `embed_tokens.npy`: (129280, 7168) — token embeddings

These can be used to find which math concepts each HLE question is about:
```python
import numpy as np
concept_dirs = np.load('~/avh_math/avh_math/db/moe_sparse_cross_600b_real/concept_dirs.npy')
embed = np.load('~/avh_math/avh_math/db/moe_sparse_cross_600b_real/embed_tokens.npy')
print(concept_dirs.shape, embed.shape)
# (15104, 4, 7168), (129280, 7168)
```

For each HLE question, embed the text and find top-matching concept_dirs.
This tells us which experts would fire, hence what domain/concept the question involves.
Use this to improve piece selection in `knowledge/concept_boost.py`.

#### Step 2e: Run improved concept_boost on HLE 2500
If concept_dirs matching works, update `knowledge/concept_boost.py` to use it better.
The current issue: concept cache was giving uniform scores (no discriminative power).

Fix approach:
1. Check why scores are uniform — maybe the similarity threshold is wrong
2. Use top-k experts (k=3-5) instead of all experts
3. Only boost pieces that strongly match concept_dirs

### Step 3: Test improvements
```bash
cd /Users/motonishikoudai/.openclaw/workspace/verantyx_v6
# Test with concept_boost enabled
DISABLE_CONCEPT_BOOST=0 python3 -c "
import sys
sys.path.insert(0, '.')
from pipeline_enhanced import VerantyxV6Enhanced
import json

pipeline = VerantyxV6Enhanced(piece_db_path='pieces/piece_db.jsonl', use_concept_boost=True)
# Test 50 questions
questions = []
with open('hle_2500_eval.jsonl') as f:
    for line in f:
        questions.append(json.loads(line))
        if len(questions) >= 50: break

from core.answer_matcher import flexible_match
correct = 0
for q in questions:
    result = pipeline.run(q['question'], q.get('answer_type', ''))
    if result.get('answer') and flexible_match(result['answer'], q['answer']):
        correct += 1
print(f'Sample 50: {correct}/50 = {correct*2}%')
"
```

### Step 4: Report
Write results to: `/Users/motonishikoudai/.openclaw/workspace/verantyx_v6/AGENT_C_RESULTS.md`

Include:
- GGUF download status (00011/00013 fixed or not, and why)
- Knowledge extraction: what was done, what was found
- concept_boost improvement (if any)
- Score improvement (if measurable)

## When done, notify:
```bash
openclaw system event --text "Agent C done: GGUF/knowledge extraction complete. Check AGENT_C_RESULTS.md" --mode now
```
