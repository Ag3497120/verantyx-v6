# Gemini ARC-AGI2 Evaluation Task

You are solving ARC-AGI2 tasks. For each task, you must:

1. Study ALL training examples carefully
2. Find the transformation pattern
3. Write a Python function `transform(grid)` that implements it
4. Apply it to test input(s) and save results

## Rules
- Grid = list of lists of ints (0-9)
- Pure Python only, NO numpy, NO external libraries
- Do NOT hardcode values from specific examples
- The function must GENERALIZE to unseen inputs
- Max 2 attempts per task, then fall back to direct prediction

## For each task_id in the batch:

### Step 1: Read the task
```python
import json
with open(f'/private/tmp/arc-agi-2/data/evaluation/{task_id}.json') as f:
    task = json.load(f)
```

### Step 2: Write transform function
Save to `~/verantyx_v6/gemini_synth_results/{task_id}.py`

The file MUST contain:
```python
def transform(grid):
    # Your implementation
    return output_grid
```

### Step 3: Verify against ALL training examples
```python
for i, ex in enumerate(task['train']):
    result = transform(ex['input'])
    if result != ex['output']:
        print(f"FAIL train {i}")
        # Fix and retry
```

### Step 4: If synth fails after 2 attempts, save direct prediction
Study the pattern and predict the test output directly.
Save to `~/verantyx_v6/gemini_direct_results/{task_id}.json`:
```json
{"test": [{"output": [[...], ...]}]}
```

### Step 5: ALWAYS save a result file (synth or direct) before moving to next task

## CRITICAL
- Save results IMMEDIATELY after each task (don't batch saves)
- Create directories if needed: `os.makedirs(..., exist_ok=True)`
- If stuck on a task for >90 seconds, save direct prediction and move on
