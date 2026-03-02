"""
build_arc_feature_router.py — ARC grid特徴量ベースのソルバールーター

embed_tokensベースの600B活性化は識別力不足。
代わりにARCタスクの構造的特徴量を直接数値化 → k-NNでソルバー推薦。

特徴量 (per train example, then aggregated):
  - input/output grid size (h, w)
  - size ratio (oh/ih, ow/iw)
  - n_colors (input, output)
  - color set overlap
  - symmetry flags (h, v, diag, rot90, rot180)
  - sparsity (non-bg ratio)
  - object count approx
  - pattern repetition indicators
  - unique color histogram
"""

import json
import os
import re
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from scipy.ndimage import label as scipy_label


def extract_grid_features(grid) -> dict:
    """Single grid → feature dict"""
    g = np.array(grid)
    h, w = g.shape
    
    colors = set(g.flatten())
    n_colors = len(colors)
    
    # Background (most common color)
    color_counts = Counter(g.flatten())
    bg = color_counts.most_common(1)[0][0]
    
    # Sparsity
    total = h * w
    non_bg = sum(1 for c in g.flatten() if c != bg)
    sparsity = non_bg / total if total > 0 else 0
    
    # Symmetry
    sym_h = float(np.array_equal(g, g[:, ::-1]))
    sym_v = float(np.array_equal(g, g[::-1, :]))
    sym_rot180 = float(np.array_equal(g, np.rot90(g, 2)))
    sym_diag = float(h == w and np.array_equal(g, g.T))
    sym_rot90 = float(h == w and np.array_equal(g, np.rot90(g)))
    
    # Object count (connected components of non-bg)
    binary = (g != bg).astype(int)
    if binary.any():
        labeled, n_objects = scipy_label(binary)
    else:
        n_objects = 0
    
    # Color histogram (normalized, 10 bins for colors 0-9)
    hist = np.zeros(10)
    for c, count in color_counts.items():
        if 0 <= c < 10:
            hist[c] = count / total
    
    return {
        "h": h, "w": w,
        "n_colors": n_colors,
        "sparsity": sparsity,
        "sym_h": sym_h, "sym_v": sym_v,
        "sym_rot180": sym_rot180, "sym_diag": sym_diag, "sym_rot90": sym_rot90,
        "n_objects": n_objects,
        "hist": hist,
    }


def task_to_feature_vector(task_data) -> np.ndarray:
    """ARC task → fixed-size feature vector"""
    features = []
    
    for ex in task_data.get("train", [])[:4]:  # max 4 examples
        inp_f = extract_grid_features(ex["input"])
        out_f = extract_grid_features(ex["output"])
        
        # Size features
        features.extend([
            inp_f["h"], inp_f["w"],
            out_f["h"], out_f["w"],
            out_f["h"] / max(inp_f["h"], 1),  # height ratio
            out_f["w"] / max(inp_f["w"], 1),  # width ratio
            float(inp_f["h"] == out_f["h"] and inp_f["w"] == out_f["w"]),  # same size
        ])
        
        # Color features
        features.extend([
            inp_f["n_colors"], out_f["n_colors"],
            out_f["n_colors"] - inp_f["n_colors"],  # color change
        ])
        
        # Structure features
        features.extend([
            inp_f["sparsity"], out_f["sparsity"],
            inp_f["sym_h"], inp_f["sym_v"], inp_f["sym_rot180"],
            inp_f["sym_diag"], inp_f["sym_rot90"],
            out_f["sym_h"], out_f["sym_v"], out_f["sym_rot180"],
            inp_f["n_objects"], out_f["n_objects"],
        ])
        
        # Histograms
        features.extend(inp_f["hist"].tolist())
        features.extend(out_f["hist"].tolist())
    
    # Pad to fixed size (4 examples * feature_per_example)
    features_per_ex = 7 + 3 + 12 + 20  # = 42
    target_len = 4 * features_per_ex
    while len(features) < target_len:
        features.append(0.0)
    
    return np.array(features[:target_len], dtype=np.float32)


def main():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    import pickle

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
    print(f"Solved: {len(solved_tasks)}")

    # Compute features for all tasks
    feature_vecs = {}
    for i, (tid, task_data) in enumerate(tasks.items()):
        if i % 100 == 0:
            print(f"  [{i}/{len(tasks)}]...")
        feature_vecs[tid] = task_to_feature_vector(task_data)

    # Analyze feature diversity
    X = np.array([feature_vecs[tid] for tid in tasks.keys()])
    print(f"\nFeature matrix: {X.shape}")
    print(f"Feature std per dim (first 20): {X.std(axis=0)[:20].round(2)}")

    # Build k-NN classifier
    # Labels: rule name for solved, "unsolved" for unsolved
    task_ids = list(tasks.keys())
    labels = [solved_tasks.get(tid, "unsolved") for tid in task_ids]
    
    # Normalize rules to base types
    def normalize_rule(rule):
        if rule == "unsolved":
            return rule
        if rule.startswith("composite"):
            return "composite"
        if rule.startswith("cross:") or rule.startswith("cross_"):
            return "cross"
        return rule.split("(")[0]
    
    labels_normalized = [normalize_rule(l) for l in labels]
    
    # Count rule types
    rule_counts = Counter(labels_normalized)
    print(f"\nRule distribution:")
    for rule, count in rule_counts.most_common(20):
        print(f"  {rule}: {count}")

    # Train k-NN (solved only as positive class)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # For each unsolved task, find its k nearest solved tasks and their rules
    solved_mask = np.array([tid in solved_tasks for tid in task_ids])
    X_solved = X_scaled[solved_mask]
    rules_solved = [labels_normalized[i] for i in range(len(task_ids)) if task_ids[i] in solved_tasks]
    
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn.fit(X_solved)
    
    # Test: for solved tasks, do leave-one-out prediction
    correct = 0
    total_solved = 0
    for i, tid in enumerate(task_ids):
        if tid not in solved_tasks:
            continue
        total_solved += 1
        # Find 6 neighbors (including self), skip self
        dists, indices = nn.kneighbors(X_scaled[i:i+1], n_neighbors=6)
        neighbor_rules = [rules_solved[idx] for idx in indices[0]]
        # Remove self (distance ~0)
        neighbor_rules_filtered = [r for j, r in enumerate(neighbor_rules) if dists[0][j] > 0.01]
        if not neighbor_rules_filtered:
            continue
        predicted = Counter(neighbor_rules_filtered).most_common(1)[0][0]
        actual = normalize_rule(solved_tasks[tid])
        if predicted == actual:
            correct += 1
    
    print(f"\nLeave-one-out accuracy: {correct}/{total_solved} ({correct/max(total_solved,1):.1%})")

    # Save model
    cache_dir = Path(os.path.expanduser("~/verantyx_v6/knowledge/expert_cluster_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "solved_features": X_solved.tolist(),
        "solved_rules": rules_solved,
        "solved_task_ids": [tid for tid in task_ids if tid in solved_tasks],
    }
    with open(cache_dir / "knn_router.json", "w") as f:
        json.dump(model_data, f)
    print(f"Saved kNN router to {cache_dir / 'knn_router.json'}")

    # Example: predict for some unsolved tasks
    print("\n=== Predictions for unsolved tasks (first 10) ===")
    count = 0
    for i, tid in enumerate(task_ids):
        if tid in solved_tasks:
            continue
        dists, indices = nn.kneighbors(X_scaled[i:i+1], n_neighbors=5)
        neighbor_rules = [rules_solved[idx] for idx in indices[0]]
        predicted = Counter(neighbor_rules).most_common(1)[0][0]
        neighbor_tids = [model_data["solved_task_ids"][idx] for idx in indices[0]]
        print(f"  {tid} → {predicted} (neighbors: {neighbor_rules}, dists: {dists[0].round(2)})")
        count += 1
        if count >= 10:
            break


if __name__ == "__main__":
    main()
