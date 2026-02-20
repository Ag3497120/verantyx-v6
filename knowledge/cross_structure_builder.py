"""
Cross Structure Builder
=======================
Converts ALL DeepSeek V3 671B MoE expert weight matrices into a
Cross DB structure (static analysis, no inference).

Usage
-----
# Full run on GPU instance
python cross_structure_builder.py \
    --model-path /workspace/models/deepseek-v3-fp8/ \
    --output /workspace/output/cross_db.json \
    --checkpoint /workspace/output/cross_db_checkpoint.json

# Resume interrupted run
python cross_structure_builder.py \
    --model-path /workspace/models/deepseek-v3-fp8/ \
    --output /workspace/output/cross_db.json \
    --checkpoint /workspace/output/cross_db_checkpoint.json \
    --resume

# Quick test: first 3 layers only
python cross_structure_builder.py \
    --model-path /workspace/models/deepseek-v3-fp8/ \
    --max-layers 3

# Local stub test (no GPU, no model)
python cross_structure_builder.py --stub
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Optional deps ──────────────────────────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    print("WARNING: torch not installed. GPU acceleration unavailable.")

try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("WARNING: safetensors not installed.  pip install safetensors")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ── Constants ──────────────────────────────────────────────────────────────────
NUM_LAYERS    = 61
NUM_EXPERTS   = 256
HIDDEN_SIZE   = 7168
FFN_DIM       = 18432
SVD_RANK      = 32          # truncated SVD: keep top-32 singular values
SVD_NITER     = 4           # power-iteration steps for svd_lowrank
CONCEPT_DIRS  = 4           # top-N left singular vectors to store
CHECKPOINT_EVERY = 10       # save checkpoint every N files


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pbar(iterable, **kwargs):
    """Wrap with tqdm if available, else plain iteration."""
    if TQDM_AVAILABLE:
        return tqdm(iterable, **kwargs)
    return iterable


def _ts() -> str:
    """ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fp8_dequantize(
    weight_tensor,                  # torch.Tensor, dtype float8*
    scale_tensor: Optional[Any],    # torch.Tensor or None
    device: str,
) -> "torch.Tensor":
    """
    Convert FP8 weight to float32.

    DeepSeek V3 stores weights as float8_e4m3fn with a companion
    *_scale_inv tensor (per-column or per-tensor).  If no scale is
    available we do a simple dtype cast (SVD ratios are scale-invariant).
    """
    W = weight_tensor.to(torch.float32)
    if scale_tensor is not None:
        s = scale_tensor.to(torch.float32).to(device)
        W = W.to(device)
        # scale_inv has shape [out_dim, 1] or scalar – broadcast safely
        try:
            W = W / s
        except RuntimeError:
            pass   # shape mismatch → skip scaling
    return W.to(device)


# ── Core SVD analysis ──────────────────────────────────────────────────────────

def analyze_weight_matrix(
    W: "torch.Tensor",
    rank: int = SVD_RANK,
    niter: int = SVD_NITER,
) -> Dict[str, Any]:
    """
    Run truncated SVD on W and extract Cross features.

    Parameters
    ----------
    W : torch.Tensor
        Weight matrix [out_dim, in_dim] in float32.
    rank : int
        Number of singular values/vectors to keep.
    niter : int
        Power iteration count for svd_lowrank.

    Returns
    -------
    dict with keys:
        features         – scalar statistics
        concept_dirs_raw – list[list[float]], shape [CONCEPT_DIRS, in_dim]
        singular_values  – list[float] (top-`rank`)
    """
    # Ensure float32 on the right device
    if W.dtype != torch.float32:
        W = W.to(torch.float32)

    # ── Truncated SVD ──────────────────────────────────────────────────────────
    # torch.svd_lowrank returns (U, S, V) where:
    #   U  [out_dim, rank]   left  singular vectors
    #   S  [rank]            singular values (descending)
    #   V  [in_dim,  rank]   right singular vectors
    #
    # concept_directions = V[:, 0:CONCEPT_DIRS].T  →  [CONCEPT_DIRS, in_dim]
    # This represents the principal *input-space* directions.
    try:
        U, S, V = torch.svd_lowrank(W, q=rank, niter=niter)
    except Exception:
        # Fallback: full SVD (slow but safe)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        S  = S[:rank]
        U  = U[:, :rank]
        V  = Vh[:rank, :].T   # convert Vh [rank, in] → V [in, rank]

    S_np = S.detach().float().cpu().numpy()
    S_np = np.maximum(S_np, 1e-12)   # numerical safety

    # ── Spectral features ──────────────────────────────────────────────────────
    p               = S_np / S_np.sum()
    spectral_entropy  = float(-np.sum(p * np.log(p + 1e-12)))
    effective_rank    = float(math.exp(spectral_entropy))
    top_singular_ratio = float(S_np[0] / S_np[1]) if S_np[1] > 1e-12 else float(S_np[0] / 1e-12)

    # ── Weight statistics ──────────────────────────────────────────────────────
    frobenius_norm  = float(torch.norm(W, p="fro").item())
    mean_abs        = float(W.abs().mean().item())
    sparsity        = float((W.abs() < 1e-5).float().mean().item())

    # ── Concept directions: top-CONCEPT_DIRS right singular vectors ───────────
    # V has shape [in_dim, rank].  V[:, :CONCEPT_DIRS].T → [CONCEPT_DIRS, in_dim]
    concept_dirs_t = V[:, :CONCEPT_DIRS].T.detach().float().cpu()
    concept_dirs   = concept_dirs_t.tolist()   # list[list[float]]

    return {
        "features": {
            "spectral_entropy":   round(spectral_entropy,   6),
            "top_singular_ratio": round(top_singular_ratio, 6),
            "effective_rank":     round(effective_rank,     6),
            "frobenius_norm":     round(frobenius_norm,     6),
            "mean_abs":           round(mean_abs,           8),
            "sparsity":           round(sparsity,           6),
        },
        "concept_dirs_raw": concept_dirs,
        "singular_values":  S_np[:rank].tolist(),
    }


# ── Coordinate normalisation ───────────────────────────────────────────────────

def compute_cross_coords(
    experts: Dict[str, Dict],
    num_layers: int,
) -> None:
    """
    In-place: add `cross_coords` to each expert entry.

    Coordinate definitions
    ----------------------
    X (abstraction) = effective_rank / max_effective_rank
                      High → expert covers many directions (broad/abstract)
                      Low  → rank-1 dominant (specialised/concrete)
    Y (application)  = log-normalised top_singular_ratio
                      High → single dominant pattern (sharp, applied skill)
                      Low  → diffuse patterns  (general background)
    Z (depth)        = layer / (num_layers - 1)
                      0 = shallowest layer, 1 = deepest
    """
    all_eff_rank = [e["features"]["effective_rank"]     for e in experts.values()]
    all_tsr      = [e["features"]["top_singular_ratio"] for e in experts.values()]

    max_eff_rank = max(all_eff_rank) if all_eff_rank else 1.0
    # Log-scale for Y: maps ratio=1→0, ratio~100→1
    max_log_tsr  = math.log1p(max(all_tsr) - 1.0 + 1e-9) if all_tsr else 1.0
    max_log_tsr  = max(max_log_tsr, 1e-9)

    for key, expert in experts.items():
        feats  = expert["features"]
        layer  = expert["layer"]

        x = float(np.clip(feats["effective_rank"] / max_eff_rank, 0.0, 1.0))
        y = float(np.clip(
            math.log1p(max(feats["top_singular_ratio"] - 1.0, 0.0)) / max_log_tsr,
            0.0, 1.0
        ))
        z = float(layer / max(num_layers - 1, 1))

        expert["cross_coords"] = [round(x, 6), round(y, 6), round(z, 6)]


# ── Safetensors scanning ───────────────────────────────────────────────────────

# Pattern: model.layers.{layer}.mlp.experts.{expert}.{proj}.weight
_EXPERT_WEIGHT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)
_SCALE_SUFFIX = ".weight_scale_inv"


def scan_safetensors_file(filepath: str) -> Dict[str, List[str]]:
    """
    Return {(layer_str, expert_str, proj): key} for all expert weights in file.
    Opens and immediately closes (no weights loaded yet).
    """
    result: Dict[str, str] = {}
    with safe_open(filepath, framework="pt", device="cpu") as f:
        for key in f.keys():
            m = _EXPERT_WEIGHT_RE.match(key)
            if m:
                tag = f"{m.group(1)}:{m.group(2)}:{m.group(3)}"
                result[tag] = key
    return result


def load_expert_weight(
    sf_handle,              # open safe_open handle
    weight_key: str,
    device: str,
) -> "torch.Tensor":
    """Load and dequantize a single expert weight tensor."""
    W_raw = sf_handle.get_tensor(weight_key)

    # Check for FP8 scale tensor
    scale_key = weight_key + "_scale_inv"
    scale = None
    try:
        keys = list(sf_handle.keys())
        if scale_key in keys:
            scale = sf_handle.get_tensor(scale_key)
    except Exception:
        pass

    return _fp8_dequantize(W_raw, scale, device)


# ── Stub mode ──────────────────────────────────────────────────────────────────

def build_stub_db(n_experts: int = 100, num_layers: int = NUM_LAYERS) -> Dict:
    """Generate a random Cross DB with `n_experts` entries for testing."""
    print(f"[STUB] Generating random Cross DB with {n_experts} experts ...")
    experts: Dict[str, Dict] = {}

    rng = random.Random(42)
    for i in range(n_experts):
        layer     = rng.randint(0, num_layers - 1)
        expert_id = rng.randint(0, NUM_EXPERTS - 1)
        key       = f"L{layer}E{expert_id}"
        if key in experts:
            expert_id = (expert_id + i) % NUM_EXPERTS
            key = f"L{layer}E{expert_id}"

        eff_rank  = rng.uniform(1.0, 32.0)
        tsr       = rng.uniform(1.1, 50.0)
        s_entropy = math.log(eff_rank)

        experts[key] = {
            "layer":     layer,
            "expert_id": expert_id,
            "features": {
                "spectral_entropy":   round(s_entropy, 6),
                "top_singular_ratio": round(tsr, 6),
                "effective_rank":     round(eff_rank, 6),
                "frobenius_norm":     round(rng.uniform(100.0, 2000.0), 3),
                "mean_abs":           round(rng.uniform(0.005, 0.05), 8),
                "sparsity":           round(rng.uniform(0.0, 0.15), 6),
            },
            "concept_directions": [
                [rng.gauss(0, 0.01) for _ in range(HIDDEN_SIZE)]
                for _ in range(CONCEPT_DIRS)
            ],
        }

    compute_cross_coords(experts, num_layers)
    return experts


# ── Main pipeline ──────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str) -> Dict:
    """Load partial results from checkpoint file."""
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            data = json.load(f)
        print(f"[RESUME] Loaded {len(data.get('experts', {}))} experts from checkpoint")
        return data
    return {"experts": {}, "processed_files": []}


def save_checkpoint(ckpt_path: str, experts: Dict, processed_files: List[str]) -> None:
    """Atomically save checkpoint."""
    tmp = ckpt_path + ".tmp"
    payload = {
        "checkpoint_at": _ts(),
        "processed_files": processed_files,
        "experts": experts,
    }
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, ckpt_path)
    print(f"[CKPT] Saved checkpoint: {len(experts)} experts, {len(processed_files)} files done")


def build_cross_db(
    model_path: str,
    output_path: str,
    checkpoint_path: str,
    resume: bool     = False,
    max_layers: Optional[int] = None,
    device: str      = "cuda",
) -> Dict:
    """
    Main pipeline: scan safetensors, extract features, build Cross DB.

    Parameters
    ----------
    model_path      : directory containing *.safetensors files
    output_path     : path for final cross_db.json
    checkpoint_path : path for incremental checkpoint JSON
    resume          : if True, skip already-processed files
    max_layers      : only process layers < max_layers (None = all)
    device          : torch device string

    Returns
    -------
    Full cross_db dict (metadata + experts)
    """
    if not SAFETENSORS_AVAILABLE:
        raise RuntimeError("safetensors not installed.  pip install safetensors")
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch not installed.  pip install torch")

    effective_num_layers = max_layers if max_layers is not None else NUM_LAYERS

    # ── Locate safetensors files ───────────────────────────────────────────────
    st_files = sorted(
        str(p) for p in Path(model_path).glob("*.safetensors")
        if not p.name.endswith(".index.json")
    )
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")
    print(f"[SCAN] Found {len(st_files)} safetensors files in {model_path}")

    # ── Resume state ───────────────────────────────────────────────────────────
    experts: Dict[str, Dict] = {}
    processed_files: List[str] = []
    if resume:
        ckpt = load_checkpoint(checkpoint_path)
        experts = ckpt.get("experts", {})
        processed_files = ckpt.get("processed_files", [])

    processed_set = set(processed_files)

    # ── GPU setup ──────────────────────────────────────────────────────────────
    if device == "cuda" and not CUDA_AVAILABLE:
        print("[WARN] CUDA not available, falling back to CPU")
        device = "cpu"

    if device == "cuda" and torch.cuda.device_count() > 1:
        print(f"[INFO] {torch.cuda.device_count()} GPUs detected; using device 0")

    print(f"[INFO] Processing device: {device}")
    print(f"[INFO] SVD rank: {SVD_RANK}  | concept_dirs: {CONCEPT_DIRS}")

    # ── Counters ───────────────────────────────────────────────────────────────
    total_experts_target = effective_num_layers * NUM_EXPERTS
    t_start = time.time()

    # ── File loop ──────────────────────────────────────────────────────────────
    files_to_process = [f for f in st_files if f not in processed_set]
    print(f"[INFO] Files to process: {len(files_to_process)} "
          f"(skipping {len(processed_set)} already done)")

    outer_bar = _pbar(
        enumerate(files_to_process),
        total=len(files_to_process),
        desc="Files",
        unit="file",
        position=0,
    )

    files_done_this_run = 0

    for file_idx, filepath in outer_bar:
        fname = os.path.basename(filepath)

        # ── Scan which expert keys live in this file ───────────────────────────
        try:
            key_map = scan_safetensors_file(filepath)
        except Exception as e:
            print(f"\n[SKIP] {fname}: scan failed – {e}")
            continue

        if not key_map:
            # No expert weights in this file
            processed_files.append(filepath)
            continue

        # Filter by max_layers
        if max_layers is not None:
            key_map = {
                tag: key for tag, key in key_map.items()
                if int(tag.split(":")[0]) < max_layers
            }

        # Group by (layer, expert_id) – we only need gate_proj
        gate_keys: Dict[Tuple[int, int], str] = {}
        for tag, key in key_map.items():
            layer_s, expert_s, proj = tag.split(":")
            if proj == "gate_proj":
                gate_keys[(int(layer_s), int(expert_s))] = key

        if not gate_keys:
            processed_files.append(filepath)
            continue

        # ── Open safetensors file once; iterate experts ────────────────────────
        with safe_open(filepath, framework="pt", device="cpu") as sf:
            inner_bar = _pbar(
                sorted(gate_keys.items()),
                desc=f"  {fname[:40]}",
                unit="exp",
                leave=False,
                position=1,
            )

            for (layer, expert_id), weight_key in inner_bar:
                exp_key = f"L{layer}E{expert_id}"
                if exp_key in experts:
                    continue   # already computed (resume)

                try:
                    W = load_expert_weight(sf, weight_key, device)
                except Exception as e:
                    if TQDM_AVAILABLE:
                        inner_bar.write(f"  [SKIP] {exp_key}: load failed – {e}")
                    else:
                        print(f"  [SKIP] {exp_key}: load failed – {e}")
                    continue

                try:
                    result = analyze_weight_matrix(W, rank=SVD_RANK, niter=SVD_NITER)
                except Exception as e:
                    print(f"  [SKIP] {exp_key}: SVD failed – {e}")
                    continue
                finally:
                    del W
                    if device == "cuda":
                        torch.cuda.empty_cache()

                experts[exp_key] = {
                    "layer":              layer,
                    "expert_id":          expert_id,
                    "features":           result["features"],
                    "concept_directions": result["concept_dirs_raw"],
                }

        processed_files.append(filepath)
        files_done_this_run += 1

        # ── Checkpoint every N files ───────────────────────────────────────────
        if files_done_this_run % CHECKPOINT_EVERY == 0:
            save_checkpoint(checkpoint_path, experts, processed_files)

        # ETA estimate
        elapsed   = time.time() - t_start
        avg_per_f = elapsed / max(files_done_this_run, 1)
        remaining = (len(files_to_process) - file_idx - 1) * avg_per_f
        if TQDM_AVAILABLE and hasattr(outer_bar, "set_postfix"):
            outer_bar.set_postfix(
                experts=len(experts),
                eta=f"{remaining/60:.1f}m",
            )

    # ── Final checkpoint ───────────────────────────────────────────────────────
    save_checkpoint(checkpoint_path, experts, processed_files)
    print(f"\n[DONE] Analysed {len(experts)} experts across {len(processed_files)} files")

    # ── Normalise Cross coordinates ────────────────────────────────────────────
    print("[POST] Computing Cross coordinates ...")
    compute_cross_coords(experts, effective_num_layers)

    # ── Assemble final DB ──────────────────────────────────────────────────────
    cross_db = {
        "metadata": {
            "model":                  "deepseek-v3-671b",
            "num_layers":             effective_num_layers,
            "num_experts":            NUM_EXPERTS,
            "total_experts_analyzed": len(experts),
            "svd_rank":               SVD_RANK,
            "concept_dirs_per_expert": CONCEPT_DIRS,
            "cross_space": {
                "X": "abstraction (effective_rank, 0=specialised → 1=broad)",
                "Y": "application  (top_singular_ratio, 0=diffuse  → 1=sharp)",
                "Z": "depth        (layer index,        0=shallow   → 1=deep)",
            },
            "created_at": _ts(),
            "model_path": model_path,
        },
        "experts": experts,
    }

    return cross_db


# ── Output helpers ─────────────────────────────────────────────────────────────

def save_cross_db(cross_db: Dict, output_path: str) -> None:
    """Write cross_db.json atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp = output_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cross_db, f, separators=(",", ":"))   # compact for large file
    os.replace(tmp, output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[SAVE] cross_db.json written: {output_path}  ({size_mb:.1f} MB)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Cross DB from DeepSeek V3 MoE expert weights",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-path", default="/workspace/models/deepseek-v3-fp8/",
        help="Directory containing *.safetensors files",
    )
    p.add_argument(
        "--output", default="/workspace/output/cross_db.json",
        help="Output path for cross_db.json",
    )
    p.add_argument(
        "--checkpoint", default="/workspace/output/cross_db_checkpoint.json",
        help="Path for incremental checkpoint file",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume from existing checkpoint",
    )
    p.add_argument(
        "--max-layers", type=int, default=None,
        help="Only process first N layers (for quick testing)",
    )
    p.add_argument(
        "--stub", action="store_true",
        help="Generate random Cross DB with 100 experts (no GPU/model needed)",
    )
    p.add_argument(
        "--stub-experts", type=int, default=100,
        help="Number of random experts to generate in stub mode",
    )
    p.add_argument(
        "--device", default="cuda",
        help="Torch device: 'cuda' or 'cpu'",
    )
    p.add_argument(
        "--svd-rank", type=int, default=SVD_RANK,
        help="Number of singular values to retain in truncated SVD",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Override global constants from CLI
    global SVD_RANK
    SVD_RANK = args.svd_rank

    print("=" * 60)
    print("  Cross Structure Builder  –  DeepSeek V3 671B MoE")
    print("=" * 60)

    if args.stub:
        # ── Stub mode ──────────────────────────────────────────────────────────
        n_layers  = args.max_layers if args.max_layers else NUM_LAYERS
        experts   = build_stub_db(n_experts=args.stub_experts, num_layers=n_layers)

        cross_db = {
            "metadata": {
                "model":                   "deepseek-v3-671b",
                "num_layers":              n_layers,
                "num_experts":             NUM_EXPERTS,
                "total_experts_analyzed":  len(experts),
                "svd_rank":                SVD_RANK,
                "concept_dirs_per_expert": CONCEPT_DIRS,
                "stub_mode":               True,
                "cross_space": {
                    "X": "abstraction",
                    "Y": "application",
                    "Z": "depth",
                },
                "created_at": _ts(),
            },
            "experts": experts,
        }

        out = args.output
        if out == "/workspace/output/cross_db.json":
            # In stub mode default to local path
            out = "cross_db_stub.json"
        save_cross_db(cross_db, out)
        print(f"[STUB] Done.  {len(experts)} experts → {out}")
        return

    # ── Real mode ──────────────────────────────────────────────────────────────
    t0 = time.time()
    cross_db = build_cross_db(
        model_path      = args.model_path,
        output_path     = args.output,
        checkpoint_path = args.checkpoint,
        resume          = args.resume,
        max_layers      = args.max_layers,
        device          = args.device,
    )
    elapsed = time.time() - t0

    save_cross_db(cross_db, args.output)

    n = cross_db["metadata"]["total_experts_analyzed"]
    print(f"\n✓ Complete: {n} experts in {elapsed/60:.1f} min")
    print(f"  Output:   {args.output}")
    print(f"  Cross space coordinates attached to every expert entry.")


if __name__ == "__main__":
    main()
