#!/usr/bin/env python3
"""
Batch solver for ARC tasks - processes task list and saves solutions.
"""
import json
import os
import sys

# Task list
TASKS = [
    "140c817e", "14754a24", "1478ab18", "14b8e18c", "150deff5", "15113be4", 
    "15660dd6", "15663ba9", "15696249", "17829a00", "178fcbfb", "17b80ad2", 
    "17b866bd", "17cae0c1", "18286ef8", "182e5d0f", "18419cfa", "184a9768", 
    "1990f7a8", "19bb5feb", "1a07d186", "1a2e2828", "1acc24af", "1ae2feb7", 
    "1b2d62fb", "1b60fb0c", "1b631d34", "1bfc4729", "1c0d0a4b", "1c786137", 
    "1caeab9d", "1cc1f1c5", "1cd22787", "1cf80156", "1d0a4b61", "1d398264", 
    "1d3b8e40", "1e0a9b12", "1e32b0e9", "1e81d6f9", "1ec26947", "1ed8deac", 
    "1efcae17", "1f0c79e5", "1f642eb9", "1f6bcee4", "1f7b1b0c", "1f7b3ee0", 
    "1f846c50", "1f876c06"
]

DATA_DIR = "/tmp/arc-agi-2/data/training"
RESULTS_DIR = os.path.expanduser("~/verantyx_v6/synth_results")
VERIFY_SCRIPT = os.path.expanduser("~/verantyx_v6/verify_transform.py")

# Track results
solved = []
failed = []

# Already solved: 140c817e
if os.path.exists(f"{RESULTS_DIR}/140c817e.py"):
    solved.append("140c817e")
    print(f"✓ 140c817e (already solved)")

# Process remaining tasks
for task_id in TASKS[1:]:  # Skip first one
    print(f"\n{'='*60}")
    print(f"Processing {task_id}...")
    print(f"{'='*60}")
    
    # This will be filled in manually
    break

print(f"\n{'='*60}")
print(f"Summary: {len(solved)} solved, {len(failed)} failed")
print(f"Solved: {solved}")
print(f"Failed: {failed}")
