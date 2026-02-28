# ARC Program Synthesis Batch Task

You are a program synthesis agent. For each ARC puzzle task, write a Python `transform(grid)` function.

## Rules
1. Read each task JSON from `/tmp/arc-agi-2/data/training/{tid}.json`
2. Study ALL train examples to find the transformation pattern
3. Write `def transform(grid):` that takes `list[list[int]]` and returns `list[list[int]]`
4. Save code to `~/verantyx_v6/synth_results/{tid}.py`
5. Verify using: `python3 ~/verantyx_v6/verify_transform.py /tmp/arc-agi-2/data/training/{tid}.json ~/verantyx_v6/synth_results/{tid}.py`
6. Only keep solutions that pass ALL train examples
7. Move to next task even if one fails
8. You may use numpy, scipy, collections
9. Do NOT hardcode outputs — find the general rule
10. Do NOT output the grid directly — always write a transform function

## Important
- Speed matters. Don't overthink. If a task looks too complex after 2 attempts, skip it.
- Track results in `~/verantyx_v6/synth_results/batch_log.json`
