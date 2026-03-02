"""
build_solver_map_v2.py — ARC構造特徴 → Expert活性化 → クラスタ → ソルバーマップ

問題: grid数値(0-9)だとembed_tokensが区別できない
解決: ARCタスクの構造的特徴を自然言語に変換してからExpert活性化を計算

特徴:
  - grid size (NxM)
  - number of unique colors
  - symmetry type (horizontal/vertical/diagonal/none)  
  - size change (input→output)
  - object count
  - pattern type descriptors
"""

import json
import os
import re
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter


def extract_features(task_data) -> str:
    """ARCタスクの構造的特徴を自然言語テキストに変換"""
    parts = []
    
    for i, ex in enumerate(task_data.get("train", [])):
        inp = ex["input"]
        out = ex["output"]
        
        ih, iw = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        
        # Colors
        inp_colors = set(c for row in inp for c in row)
        out_colors = set(c for row in out for c in row)
        
        # Size relationship
        if oh == ih and ow == iw:
            size_rel = "same size transformation"
        elif oh > ih or ow > iw:
            size_rel = f"expansion scaling upscale grow {oh/ih:.1f}x"
        else:
            size_rel = f"reduction crop shrink extract {oh/ih:.1f}x"
        
        # Symmetry check
        syms = []
        inp_np = np.array(inp)
        if np.array_equal(inp_np, inp_np[::-1]):
            syms.append("vertical symmetry mirror flip")
        if np.array_equal(inp_np, inp_np[:, ::-1]):
            syms.append("horizontal symmetry mirror flip")
        if ih == iw and np.array_equal(inp_np, inp_np.T):
            syms.append("diagonal symmetry transpose")
            
        # Color mapping
        new_colors = out_colors - inp_colors
        removed_colors = inp_colors - out_colors
        
        # Object detection (connected components approximation)
        # Count distinct non-background regions
        bg = Counter(c for row in inp for c in row).most_common(1)[0][0]
        non_bg_count = sum(1 for row in inp for c in row if c != bg)
        
        # Repetition/tiling check
        tiling = ""
        if oh == ih * 2 or oh == ih * 3 or ow == iw * 2 or ow == iw * 3:
            tiling = "tiling repetition copy duplicate pattern"
        
        # Build text
        desc = (
            f"grid {ih}x{iw} to {oh}x{ow} "
            f"{len(inp_colors)} colors {len(out_colors)} output colors "
            f"{size_rel} "
            f"{' '.join(syms) if syms else 'asymmetric'} "
            f"{'color mapping recolor' if new_colors else ''} "
            f"{'remove filter' if removed_colors else ''} "
            f"non-background pixels {non_bg_count} "
            f"{tiling} "
            f"{'flood fill region' if non_bg_count > 10 else 'sparse objects'} "
        )
        parts.append(desc)
    
    return " ".join(parts)


def main():
    from knowledge.expert_cluster import ExpertClusterRouter

    task_dir = Path("/tmp/arc-agi-2/data/training")
    tasks = {}
    for f in sorted(task_dir.glob("*.json")):
        with open(f) as fh:
            tasks[f.stem] = json.load(fh)
    print(f"Loaded {len(tasks)} tasks")

    # Parse solved tasks
    log_path = Path(os.path.expanduser("~/verantyx_v6/arc_cross_engine_v9.log"))
    solved_tasks = {}
    with open(log_path) as f:
        for line in f:
            m = re.search(r'✓.*?([0-9a-f]{8})\s+ver=\d+\s+rule=(.+)', line)
            if m:
                solved_tasks[m.group(1)] = m.group(2).strip()
    
    # Also add synth results from the latest scoring
    score_file = Path(os.path.expanduser("~/verantyx_v6/training_score_detail.json"))
    if score_file.exists():
        with open(score_file) as f:
            detail = json.load(f)
        for tid, info in detail.items():
            if info.get("correct") and tid not in solved_tasks:
                solved_tasks[tid] = info.get("solver", "synth")

    print(f"Solved: {len(solved_tasks)}")

    router = ExpertClusterRouter()

    # Per-cluster stats  
    cluster_rule_solved = defaultdict(lambda: defaultdict(int))
    cluster_total = defaultdict(int)
    cluster_solved_count = defaultdict(int)
    
    # Store per-task cluster signature for analysis
    task_clusters = {}

    for i, (tid, task_data) in enumerate(tasks.items()):
        if i % 100 == 0:
            print(f"  [{i}/{len(tasks)}]...")

        text = extract_features(task_data)
        cluster_dist = router.get_cluster_distribution(text, top_k=10)
        
        task_clusters[tid] = cluster_dist

        rule = solved_tasks.get(tid)
        
        for cid, activation in cluster_dist:
            cluster_total[cid] += 1
            if rule:
                cluster_solved_count[cid] += 1
                # Normalize rule
                base = rule.split("(")[0] if "(" in rule else rule
                base = base.replace("cross:", "cross_")
                cluster_rule_solved[cid][base] += 1

    # Check diversity
    first_clusters = [task_clusters[tid][0][0] for tid in list(tasks.keys())[:100]]
    print(f"\nFirst cluster diversity (first 100 tasks): {len(set(first_clusters))} unique clusters")
    print(f"Distribution: {Counter(first_clusters).most_common(10)}")

    # Build solver_map
    solver_map = {}
    for cid in range(router.n_clusters):
        total = cluster_total.get(cid, 0)
        solved = cluster_solved_count.get(cid, 0)
        if total == 0:
            continue
        
        entry = {
            "_total": {
                "count": total,
                "solved": solved,
                "success_rate": round(solved / total, 4),
            }
        }
        
        for rule, count in sorted(cluster_rule_solved.get(cid, {}).items(), key=lambda x: -x[1]):
            entry[rule] = {
                "count": count,
                "success_rate": round(count / total, 4),
            }
        
        solver_map[str(cid)] = entry

    # Save
    cache_dir = Path(os.path.expanduser("~/verantyx_v6/knowledge/expert_cluster_cache"))
    out_path = cache_dir / "solver_map.json"
    with open(out_path, "w") as f:
        json.dump(solver_map, f, indent=2)
    
    # Also save task_clusters for analysis
    task_clusters_serializable = {tid: [(int(c), float(s)) for c, s in dist] for tid, dist in task_clusters.items()}
    with open(cache_dir / "task_clusters.json", "w") as f:
        json.dump(task_clusters_serializable, f)

    print(f"\nSaved to {out_path}")

    # Print interesting clusters
    print("\n=== Clusters with high solve rates ===")
    ranked = sorted(
        [(cid, entry) for cid, entry in solver_map.items()],
        key=lambda x: -x[1]["_total"]["success_rate"]
    )
    for cid_str, entry in ranked[:20]:
        t = entry["_total"]
        if t["count"] < 5:
            continue
        top_rules = [(k, v["count"]) for k, v in entry.items() if k != "_total"]
        top_rules.sort(key=lambda x: -x[1])
        rules_str = ", ".join(f"{r}({c})" for r, c in top_rules[:5])
        print(f"  C{int(cid_str):02d}: {t['solved']}/{t['count']} ({t['success_rate']:.1%})  {rules_str}")
    
    print("\n=== Clusters with low solve rates (opportunities) ===")
    ranked_low = sorted(
        [(cid, entry) for cid, entry in solver_map.items()],
        key=lambda x: x[1]["_total"]["success_rate"]
    )
    for cid_str, entry in ranked_low[:10]:
        t = entry["_total"]
        if t["count"] < 20:
            continue
        print(f"  C{int(cid_str):02d}: {t['solved']}/{t['count']} ({t['success_rate']:.1%})")


if __name__ == "__main__":
    main()
