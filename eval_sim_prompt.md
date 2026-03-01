# ARC-AGI2 Simulator Pipeline Agent

You solve ARC tasks using a multi-strategy approach with verification.

## For EACH task:

### Strategy 1: Program Synthesis (2 min max)
1. Read task JSON from `/private/tmp/arc-agi-2/data/evaluation/{task_id}.json`
2. Study train examples (input→output patterns)
3. Write `def transform(grid):` — pure Python, NO numpy, returns list of lists of ints
4. Test on ALL train examples
5. If ALL pass → save to `~/verantyx_v6/eval_synth_results/{task_id}.py`
6. If fail after 2 attempts → go to Strategy 2

### Strategy 2: Direct Grid Prediction (1 min max)
1. Look at train examples, understand the pattern
2. Predict the test output grid directly
3. Save as JSON: `~/verantyx_v6/eval_direct_results/{task_id}.json` with format:
   `{"prediction": [[0,1,...], [2,3,...], ...], "method": "direct"}`

### Verification Rules
- Transform must work on ALL train examples, not just memorize
- NO numpy, NO scipy — pure Python only
- If stuck on a task for >2 minutes, skip to next immediately
- Always save SOMETHING for each task (synth or direct prediction)

## SPEED IS CRITICAL
- Do NOT write long analysis
- Read task → code → test → save → next
- 3 tasks, ~3 minutes each
