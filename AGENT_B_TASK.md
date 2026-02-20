# Agent B Task: Failure Analysis + Pipeline Quick Wins

## Context
You are working on **Verantyx V6**, a rule-based reasoning system for HLE (Humanity's Last Exam).
- Workspace: `/Users/motonishikoudai/.openclaw/workspace/verantyx_v6/`
- Current bias-free score: **3.80% (95/2500)**
- Goal: Find and fix bugs/issues to raise score without adding bias
- Deadline: Feb 28, 2026 (9 days)

## Architecture Overview
- `pipeline_enhanced.py` — main pipeline (VerantyxV6Enhanced)
- `decomposer/decomposer.py` — question → IR (IntermediateRepresentation)
- `cegis/cegis_loop.py` — CEGIS loop (8 steps, generates counterexamples)
- `executors/*.py` — domain-specific solvers
- `pieces/piece_db.jsonl` — knowledge pieces (~110 pieces)
- `quick_eval_hle.py` — evaluation script

## Known Issues (fix these first)

### Issue 1: simulation_proved Bug (partially fixed)
In `pipeline_enhanced.py`, check that CrossSimulation results properly return `proved=True`.
Verify the fix is correct:
```python
# Look for CrossSimulation in pipeline_enhanced.py
# The bug was: proved results were not being returned
# Status: "fixed" but score went from 4.08% → 4.12% (small gain)
# Investigate if there are more places where proved=True is lost
```

### Issue 2: Stub Executors That Regressed Score
Implementing `algebra_solve_linear`, `algebra_factor`, `partition_number` REDUCED score from 4.12% → 3.80%.
This means these executors are now computing WRONG answers where they previously returned lucky stubs.
**Fix:** Debug these three executors and make them correct OR remove them:
- `executors/algebra.py` — `solve_linear_equation`, `factor_polynomial`, `partition_number`
- Run targeted tests to see what they output vs expected

### Issue 3: Decomposer Entity Extraction
Known gaps in decomposer (from 2026-02-20 morning session):
- `!` factorial not detected → `nt_factorial` piece not selected
- `^` power notation → entities missing `base`/`exponent`
- Permutation with single number → `r=n` not inferred

**These patches were planned but may not have been applied:**
```python
# In decomposer/decomposer.py _detect_keywords:
if '!' in text and re.search(r'\d+\s*!', text):
    keywords.append('factorial')

# In _extract_entities:
power_match = re.search(r'(\d+)\s*[\^]\s*(\d+)', text)
if power_match:
    entities['base'] = int(power_match.group(1))
    entities['exponent'] = int(power_match.group(2))
```

## Your Mission

### Step 1: Run evaluation and capture per-question results
```bash
cd /Users/motonishikoudai/.openclaw/workspace/verantyx_v6
python3 quick_eval_hle.py 2>&1 | tee /tmp/agent_b_eval_run.log
```
While running, check if the output shows per-question status.

If quick_eval_hle.py doesn't save per-question results, modify it to save to a JSON file.

### Step 2: Analyze failures
After eval, load the results and categorize failures:
```python
import json

# Load eval results (check hle_2500_current_eval.json or similar)
with open('hle_2500_current_eval.json') as f:
    results = json.load(f)

# Count by category
from collections import Counter
fail_types = Counter()
for r in results:
    if not r.get('correct'):
        fail_types[r.get('fail_reason', 'unknown')] += 1
print(fail_types.most_common(20))
```

### Step 3: Fix the regressed stub executors
Debug `executors/algebra.py` — the three functions that regressed score:
1. `solve_linear_equation` — should solve ax + b = c for x
2. `factor_polynomial` — should factor polynomials
3. `partition_number` — should compute integer partitions

For each:
1. Check what input format the pipeline sends them
2. Check what they currently output
3. Fix or revert to stub if the real implementation is wrong

### Step 4: Check decomposer patches
Verify if the planned decomposer patches were applied:
```bash
grep -n "factorial" decomposer/decomposer.py | head -20
grep -n "exponent" decomposer/decomposer.py | head -20
```

Apply any missing patches.

### Step 5: Check for false positives
The pipeline might be outputting WRONG answers for questions it "thinks" it solved.
Find cases where:
- `simulation_proved=True` but answer is wrong
- `cegis_proved=True` but answer is wrong

These are bugs in verification logic.

### Step 6: Implement improvements
Based on your analysis, implement fixes. Priority order:
1. Fix regressed algebra executors
2. Apply decomposer patches
3. Fix any false positive verification bugs
4. Add any quick-win executors for common question patterns

### Step 7: Run final eval
```bash
cd /Users/motonishikoudai/.openclaw/workspace/verantyx_v6
python3 quick_eval_hle.py 2>&1 | tee /tmp/agent_b_final_eval.log
```

### Step 8: Report results
Write to: `/Users/motonishikoudai/.openclaw/workspace/verantyx_v6/AGENT_B_RESULTS.md`

Include:
- Starting score: 3.80%
- Final score: X%
- What fixes were applied
- What bugs were found but not yet fixed (for other agents)

## Important Rules
- **NO bias** — no position_prior, no hardcoded answers
- **Regression test** — if a change drops score, revert it
- **Document all changes** in AGENT_B_RESULTS.md

## When done, notify:
```bash
openclaw system event --text "Agent B done: Pipeline fixes complete. Score: X%. Check AGENT_B_RESULTS.md" --mode now
```
