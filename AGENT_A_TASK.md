# Agent A Task: MCQ + CEGIS Enhancement

## Context
You are working on **Verantyx V6**, a rule-based reasoning system for HLE (Humanity's Last Exam).
- Workspace: `/Users/motonishikoudai/.openclaw/workspace/verantyx_v6/`
- Goal: Improve bias-free score from 3.80% (95/2500) to as high as possible
- Deadline: Feb 28, 2026 (9 days)

## Current State
- MCQ questions: 628 out of 2500 total HLE questions
- Current MCQ correct: 82/628 (13.1%)
- CEGIS proved: 67 MCQ questions (the main source of MCQ wins)
- Pipeline file: `pipeline_enhanced.py`
- CEGIS files: `cegis/cegis_loop.py`, `cegis/certificate.py`, `cegis/worldgen.py`

## Your Mission
**Maximize MCQ correctness without any position_prior bias or hardcoded answers.**

Every correct answer must come from either:
1. CEGIS verification (cegis_proved=True)
2. Direct computation by an executor (simulation_proved=True)
3. Logical deduction from pieces

### Step 1: Analyze current MCQ performance
Run the eval on a sample and capture MCQ-specific failures:
```bash
cd /Users/motonishikoudai/.openclaw/workspace/verantyx_v6
python3 -c "
import json
questions = []
with open('hle_2500_eval.jsonl') as f:
    for line in f:
        q = json.loads(line)
        if q.get('answer_type') == 'multiple_choice':
            questions.append(q)
print(f'MCQ questions: {len(questions)}')
# Show first 10
for q in questions[:5]:
    print(q['question'][:100], '|', q.get('answer', ''))
"
```

### Step 2: Find which MCQ domains CEGIS can solve
Look at `cegis/worldgen.py` — it has 11 domains. Which MCQ questions fall into these domains?

Identify:
- How many MCQ questions are in arithmetic/combinatorics/logic domains (CEGIS-covered)
- Which MCQ question types are NOT covered

### Step 3: Expand CEGIS MCQ coverage
Add new WorldGenerator domains or strategies specifically for MCQ patterns:

Common MCQ patterns in HLE:
- "Which of the following is true?" → logical elimination
- "What is the value of X?" → compute + match to options
- "Which best describes?" → domain knowledge lookup
- Multiple steps: eliminate wrong answers via computation

**Concrete improvements to implement:**

#### 3a. Better MCQ option verification
In `cegis/cegis_loop.py`, for MCQ questions, enhance the verification:
- For each option A/B/C/D/E, try to VERIFY it's true or DISPROVE it
- If exactly one option is verified → that's the answer
- Current issue: CEGIS may miss MCQ because it doesn't iterate over options

#### 3b. MCQ elimination strategy
In `executors/multiple_choice.py`:
- Add a strategy that uses other executors to check each option
- E.g., if option A says "the result is 42" and executor computes 42 → choose A

#### 3c. More CEGIS certificate types
In `cegis/certificate.py`:
- Add `MCQVerified` certificate for option-by-option verification
- Add `EliminationProof` certificate when all others are disproved

### Step 4: Test and verify (no bias!)
After each change, test:
```bash
cd /Users/motonishikoudai/.openclaw/workspace/verantyx_v6
python3 -c "
import sys
sys.path.insert(0, '.')
from pipeline_enhanced import VerantyxV6Enhanced
import json

pipeline = VerantyxV6Enhanced(piece_db_path='pieces/piece_db.jsonl')
questions = []
with open('hle_2500_eval.jsonl') as f:
    for line in f:
        q = json.loads(line)
        if q.get('answer_type') == 'multiple_choice':
            questions.append(q)

from core.answer_matcher import flexible_match
correct = 0
for q in questions[:100]:
    result = pipeline.run(q['question'], q.get('answer_type', ''))
    if result.get('answer') and flexible_match(result['answer'], q['answer']):
        correct += 1
print(f'MCQ 100-sample: {correct}/100')
"
```

### Step 5: Report results
When done, write your results to:
`/Users/motonishikoudai/.openclaw/workspace/verantyx_v6/AGENT_A_RESULTS.md`

Include:
- What you implemented
- MCQ score before/after
- Any issues found

## Important Rules
- **NO position_prior** (never use A/B/C/D frequency statistics)
- **NO hardcoded answers** (never put "if question contains X, answer Y")
- **NO general_detectors** that are question-specific memorization
- Only verified or computed answers count

## When done, notify:
```bash
openclaw system event --text "Agent A done: MCQ improvement complete. Check AGENT_A_RESULTS.md" --mode now
```
