"""
build_solver_map.py — Cross Engine 正解タスク × Expert クラスタ → solver_map.json
"""

import json
import os
import re
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict


def grid_to_text(grid):
    return "\n".join(" ".join(str(c) for c in row) for row in grid)

def task_to_text(task_data):
    parts = []
    for i, ex in enumerate(task_data.get("train", [])):
        parts.append(f"Input:\n{grid_to_text(ex['input'])}")
        parts.append(f"Output:\n{grid_to_text(ex['output'])}")
    return "\n\n".join(parts)


def main():
    from knowledge.expert_cluster import ExpertClusterRouter

    # Load ARC tasks
    task_dir = Path("/tmp/arc-agi-2/data/training")
    tasks = {}
    for f in sorted(task_dir.glob("*.json")):
        with open(f) as fh:
            tasks[f.stem] = json.load(fh)
    print(f"Loaded {len(tasks)} tasks")

    # Parse cross engine log for solved tasks + their rules
    log_path = Path(os.path.expanduser("~/verantyx_v6/arc_cross_engine_v9.log"))
    solved_tasks = {}  # tid -> rule
    with open(log_path) as f:
        for line in f:
            m = re.search(r'✓.*?([0-9a-f]{8})\s+ver=\d+\s+rule=(.+)', line)
            if m:
                tid = m.group(1)
                rule = m.group(2).strip()
                solved_tasks[tid] = rule

    print(f"Found {len(solved_tasks)} cross-engine solved tasks")

    # Also check synth solutions
    synth_dir = Path(os.path.expanduser("~/verantyx_v6/synth_solutions"))
    synth_count = 0
    if synth_dir.exists():
        for f in synth_dir.glob("*.py"):
            tid = f.stem
            if tid not in solved_tasks:
                solved_tasks[tid] = "synth:" + tid
                synth_count += 1
    
    # Check verified synth
    synth_verified = Path(os.path.expanduser("~/verantyx_v6/synth_verified.json"))
    if synth_verified.exists():
        with open(synth_verified) as f:
            sv = json.load(f)
        for tid in sv:
            if tid not in solved_tasks:
                solved_tasks[tid] = "synth:" + tid
                synth_count += 1
    
    print(f"Total solved: {len(solved_tasks)} (synth added: {synth_count})")

    # Initialize router
    router = ExpertClusterRouter()

    # Compute cluster distribution for each task
    # cluster_id -> rule -> count
    cluster_rule_solved = defaultdict(lambda: defaultdict(int))
    cluster_rule_total = defaultdict(lambda: defaultdict(int))
    cluster_total = defaultdict(int)
    cluster_solved = defaultdict(int)

    for i, (tid, task_data) in enumerate(tasks.items()):
        if i % 100 == 0:
            print(f"  [{i}/{len(tasks)}]...")

        text = task_to_text(task_data)
        cluster_dist = router.get_cluster_distribution(text, top_k=10)

        is_solved = tid in solved_tasks
        rule = solved_tasks.get(tid, "unsolved")
        
        # Normalize rule to base type
        base_rule = rule.split(":")[0] if ":" in rule else rule
        # For composite rules like composite(A+B), keep as-is but also track components
        
        for cid, activation in cluster_dist:
            cluster_total[cid] += 1
            if is_solved:
                cluster_solved[cid] += 1
                cluster_rule_solved[cid][base_rule] += 1

    # Build solver_map
    solver_map = {}
    for cid in range(router.n_clusters):
        total = cluster_total.get(cid, 0)
        solved = cluster_solved.get(cid, 0)
        if total == 0:
            continue
        
        entry = {
            "_total": {
                "count": total,
                "solved": solved,
                "success_rate": round(solved / total, 4),
            }
        }
        
        # Per-rule stats
        for rule, count in sorted(cluster_rule_solved.get(cid, {}).items(), key=lambda x: -x[1]):
            entry[rule] = {
                "count": count,
                "solved": count,
                "success_rate": round(count / total, 4),
            }
        
        solver_map[str(cid)] = entry

    # Save
    cache_dir = Path(os.path.expanduser("~/verantyx_v6/knowledge/expert_cluster_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "solver_map.json"
    with open(out_path, "w") as f:
        json.dump(solver_map, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print summary
    print("\n=== Cluster Solve Rates (top clusters by volume) ===")
    ranked = sorted(solver_map.items(), key=lambda x: -x[1]["_total"]["count"])
    for cid_str, entry in ranked[:20]:
        t = entry["_total"]
        top_rules = [(k, v["count"]) for k, v in entry.items() if k != "_total"]
        top_rules.sort(key=lambda x: -x[1])
        rules_str = ", ".join(f"{r}({c})" for r, c in top_rules[:5])
        print(f"  C{int(cid_str):02d}: {t['solved']}/{t['count']} ({t['success_rate']:.1%})  {rules_str}")


if __name__ == "__main__":
    main()
