#!/bin/bash
# =============================================================================
# Verantyx Phase B – Cross DB Builder
# =============================================================================
# Runs AFTER Phase A (expert routing / semantic extraction) has completed.
#
# Phase A output:
#   /workspace/output/pieces_600b_extracted.jsonl  ← semantic knowledge pieces
#   /workspace/verantyx_v6/knowledge/runpod_deployment/routing_cache.json
#
# Phase B output:
#   /workspace/output/cross_db.json                ← full 3D Cross DB
#   /workspace/output/cross_db_checkpoint.json     ← incremental checkpoint
#
# Recommended pod config (same as Phase A):
#   GPU: 8× H100 80GB (640 GB VRAM total)   – ~20-30 min
#        or 8× A100 80GB                     – ~45-60 min
#   RAM: 256 GB+ system RAM
#   Storage: 800 GB NVMe (model ~670 GB + outputs)
#   Image: PyTorch 2.1+ / CUDA 12.1+
#
# Usage
# -----
#   bash run_phase_b.sh [--resume] [--max-layers N] [--stub]
#
# Options
#   --resume      Resume from existing checkpoint (skip already-processed files)
#   --max-layers  Only process first N layers (quick test: --max-layers 3)
#   --stub        Generate random Cross DB without model (local test)
# =============================================================================

set -euo pipefail

# ── Parse arguments ────────────────────────────────────────────────────────────
RESUME_FLAG=""
MAX_LAYERS_FLAG=""
STUB_FLAG=""
EXTRA_ARGS=""

for arg in "$@"; do
    case "$arg" in
        --resume)       RESUME_FLAG="--resume" ;;
        --stub)         STUB_FLAG="--stub" ;;
        --max-layers)   shift ; MAX_LAYERS_FLAG="--max-layers $1" ;;
        --max-layers=*) MAX_LAYERS_FLAG="--max-layers ${arg#*=}" ;;
        *)              EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
    esac
done

# ── Config ─────────────────────────────────────────────────────────────────────
WORK_DIR="${WORK_DIR:-/workspace/verantyx_v6}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/deepseek-v3-fp8}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output}"
PYTHON="${PYTHON:-python3}"
LOG_FILE="${OUTPUT_DIR}/phase_b.log"

CROSS_DB_PATH="${OUTPUT_DIR}/cross_db.json"
CHECKPOINT_PATH="${OUTPUT_DIR}/cross_db_checkpoint.json"

# ── Banner ─────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Verantyx Phase B  –  Cross DB Builder"
echo "  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"
echo "  Work dir    : $WORK_DIR"
echo "  Model dir   : $MODEL_DIR"
echo "  Output dir  : $OUTPUT_DIR"
echo "  Cross DB    : $CROSS_DB_PATH"
echo "  Checkpoint  : $CHECKPOINT_PATH"
[[ -n "$RESUME_FLAG" ]]     && echo "  Resume      : YES"
[[ -n "$MAX_LAYERS_FLAG" ]] && echo "  $MAX_LAYERS_FLAG"
[[ -n "$STUB_FLAG" ]]       && echo "  Mode        : STUB (no model)"
echo "------------------------------------------------------------"

# ── Prerequisite checks ────────────────────────────────────────────────────────
if [[ -z "$STUB_FLAG" ]]; then
    echo "[CHECK] Verifying Phase A outputs..."

    if [[ ! -f "${OUTPUT_DIR}/pieces_600b_extracted.jsonl" ]]; then
        echo "[WARN]  Phase A output not found: pieces_600b_extracted.jsonl"
        echo "        Run Phase A first, or continue anyway? (5s timeout → continue)"
        read -t 5 -p "        Press Ctrl-C to abort, or Enter to continue: " || true
    else
        PIECE_COUNT=$(wc -l < "${OUTPUT_DIR}/pieces_600b_extracted.jsonl" || echo "?")
        echo "[OK]    Phase A pieces: ${PIECE_COUNT} lines"
    fi

    echo "[CHECK] Model directory..."
    if [[ ! -d "$MODEL_DIR" ]]; then
        echo "[ERROR] Model directory not found: $MODEL_DIR"
        echo "        Run setup.sh first (MODEL_VARIANT=fp8 bash setup.sh)"
        exit 1
    fi
    ST_COUNT=$(find "$MODEL_DIR" -name "*.safetensors" | wc -l || echo "0")
    if [[ "$ST_COUNT" -lt 1 ]]; then
        echo "[ERROR] No .safetensors files found in $MODEL_DIR"
        exit 1
    fi
    echo "[OK]    Found $ST_COUNT safetensors files"
fi

# ── GPU info ───────────────────────────────────────────────────────────────────
echo ""
echo "[INFO] GPU status:"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free \
               --format=csv,noheader,nounits | \
        awk -F',' '{printf "       GPU %s: %s  total=%s MB  free=%s MB\n", $1, $2, $3, $4}'
else
    echo "       (nvidia-smi not found)"
fi

# ── Python dependency check ────────────────────────────────────────────────────
echo ""
echo "[CHECK] Python dependencies..."
$PYTHON - <<'PYCHECK'
import sys
missing = []
for pkg in ["torch", "safetensors", "numpy", "tqdm"]:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f"[ERROR] Missing packages: {', '.join(missing)}")
    print("        Run: pip install " + " ".join(missing))
    sys.exit(1)
import torch
print(f"[OK]    torch={torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"devices={torch.cuda.device_count()}")
PYCHECK

# ── Output directory ───────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
echo ""

# ── Resume logic ───────────────────────────────────────────────────────────────
if [[ -n "$RESUME_FLAG" && -f "$CHECKPOINT_PATH" ]]; then
    CKPT_EXPERTS=$(python3 -c "
import json
d = json.load(open('${CHECKPOINT_PATH}'))
print(len(d.get('experts', {})))
" 2>/dev/null || echo "?")
    echo "[RESUME] Checkpoint has ${CKPT_EXPERTS} experts already computed"
fi

# ── Estimate runtime ───────────────────────────────────────────────────────────
if [[ -z "$STUB_FLAG" && -z "$MAX_LAYERS_FLAG" ]]; then
    GPU_COUNT=$($PYTHON -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
    echo "[INFO]  Expected runtime:"
    echo "        $GPU_COUNT× H100 80GB  →  ~20-30 minutes"
    echo "        $GPU_COUNT× A100 80GB  →  ~45-60 minutes"
    echo "        CPU only               →  ~8-12 hours"
fi

# ── Run Phase B ────────────────────────────────────────────────────────────────
echo ""
echo "[RUN]  Starting cross_structure_builder.py ..."
echo "       Log: $LOG_FILE"
echo ""

# Build the command
CMD="$PYTHON $WORK_DIR/knowledge/cross_structure_builder.py"

if [[ -n "$STUB_FLAG" ]]; then
    CMD="$CMD --stub --output ${CROSS_DB_PATH}"
else
    CMD="$CMD \
        --model-path     $MODEL_DIR \
        --output         $CROSS_DB_PATH \
        --checkpoint     $CHECKPOINT_PATH \
        --device         cuda \
        $RESUME_FLAG \
        $MAX_LAYERS_FLAG"
fi

# Echo the command for transparency
echo "CMD: $CMD"
echo ""

# Run with tee to both terminal and log
mkdir -p "$(dirname "$LOG_FILE")"
$CMD 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

# ── Post-run check ─────────────────────────────────────────────────────────────
echo ""
echo "------------------------------------------------------------"

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "[FAIL]  cross_structure_builder.py exited with code $EXIT_CODE"
    echo "        Check log: $LOG_FILE"
    exit $EXIT_CODE
fi

if [[ ! -f "$CROSS_DB_PATH" ]]; then
    echo "[FAIL]  cross_db.json not found at $CROSS_DB_PATH"
    exit 1
fi

# Count experts in output
EXPERT_COUNT=$($PYTHON -c "
import json
db = json.load(open('${CROSS_DB_PATH}'))
meta    = db.get('metadata', {})
experts = db.get('experts', {})
total   = meta.get('total_experts_analyzed', len(experts))
print(total)
" 2>/dev/null || echo "?")

FILE_SIZE=$(du -sh "$CROSS_DB_PATH" | cut -f1)

echo "[OK]    Phase B complete"
echo ""
echo "  Experts analysed : $EXPERT_COUNT"
echo "  Output file      : $CROSS_DB_PATH  ($FILE_SIZE)"
echo "  Log              : $LOG_FILE"
echo ""

# ── Quick validation ───────────────────────────────────────────────────────────
echo "[VALIDATE] Spot-checking Cross DB structure..."
$PYTHON - "$CROSS_DB_PATH" <<'PYVAL'
import sys, json, random
path = sys.argv[1]
db = json.load(open(path))
meta    = db["metadata"]
experts = db["experts"]

# Check metadata
assert "num_layers"  in meta, "missing num_layers"
assert "cross_space" in meta, "missing cross_space"

# Check a random expert
keys = list(experts.keys())
sample = random.sample(keys, min(5, len(keys)))
for k in sample:
    e = experts[k]
    assert "cross_coords"  in e, f"{k}: missing cross_coords"
    assert "features"      in e, f"{k}: missing features"
    assert len(e["cross_coords"]) == 3, f"{k}: cross_coords should have 3 values"
    feats = e["features"]
    for fkey in ["spectral_entropy","top_singular_ratio","effective_rank",
                 "frobenius_norm","mean_abs","sparsity"]:
        assert fkey in feats, f"{k}: missing feature '{fkey}'"

print(f"[OK]    Validated {len(sample)} random experts – structure looks good")
print(f"        Total: {len(keys)} experts  |  layers: {meta['num_layers']}  |  "
      f"experts/layer: {meta.get('num_experts', '?')}")
PYVAL

# ── Next steps ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Phase B Complete!"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Download Cross DB to local machine:"
echo "     scp root@<pod_ip>:${CROSS_DB_PATH} \\"
echo "         /Users/motonishikoudai/.openclaw/workspace/verantyx_v6/knowledge/"
echo ""
echo "  2. Test search locally:"
echo "     python knowledge/cross_searcher.py \\"
echo "         --db knowledge/cross_db.json \\"
echo "         --domain math --k 20"
echo ""
echo "  3. Integrate into Verantyx routing:"
echo "     # cross_mapper.py can now load cross_db.json directly"
echo "     # instead of computing from profiler output"
echo "============================================================"
