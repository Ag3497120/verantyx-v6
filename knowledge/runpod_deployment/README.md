# Verantyx 600B Knowledge Extraction - RunPod Deployment Guide

## Overview

This directory contains everything needed to run the DeepSeek V3 600B knowledge
extraction pipeline on RunPod (or any CUDA-capable GPU server).

The pipeline extracts mathematical and scientific knowledge from DeepSeek V3's
MoE expert weights using:
1. **Expert routing analysis** – probe queries reveal which experts specialize in which domains
2. **Semantic knowledge extraction** – targeted probing of top experts yields structured knowledge
3. **Piece conversion** – extracted knowledge is formatted as Verantyx `piece_db.jsonl` entries

---

## Quick Start: Stub Mode (No GPU Needed)

Run locally to test the full pipeline without a real model:

```bash
cd /Users/motonishikoudai/.openclaw/workspace/verantyx_v6
python knowledge/runpod_deployment/run_extraction.py --stub
```

Expected output:
```
[EXPERT_ROUTER] Analyzing 8 domains...
[SEMANTIC_EXTRACTOR] Extracted 147 high-confidence pieces
[PIECE_CONVERTER] Wrote 147 pieces to pieces/pieces_600b_extracted.jsonl
✓ SUCCESS: Extracted 147 pieces (≥ 100 target met)
```

---

## RunPod Deployment: Step by Step

### Step 1: Launch a RunPod Instance

1. Go to [runpod.io](https://runpod.io) → **Deploy** → **GPU Pods**
2. Select GPU configuration:
   - **Minimum**: 3× A100 80GB (for Q4 GGUF)
   - **Recommended**: 5× A100 80GB (for Q4 or FP8)
   - **Best**: 8× A100 80GB (for BF16 full precision)
3. Choose template: **RunPod PyTorch 2.1** or **CUDA 12.1+**
4. Set disk: **800 GB** (400 GB model + 400 GB buffer)
5. Click **Deploy**

### Step 2: Connect to the Instance

```bash
# Via SSH (RunPod provides the SSH command)
ssh root@<pod_ip> -p <port>

# Or use RunPod's web terminal
```

### Step 3: Upload Project Code

From your local machine:
```bash
# Upload project files (excluding venv and large data files)
rsync -avz \
    --exclude='venv/' \
    --exclude='*.log' \
    --exclude='hle_*' \
    --exclude='*.parquet' \
    /Users/motonishikoudai/.openclaw/workspace/verantyx_v6/ \
    root@<pod_ip>:/workspace/verantyx_v6/

# Or use the RunPod file browser UI
```

### Step 4: Run Setup Script

```bash
cd /workspace/verantyx_v6/knowledge/runpod_deployment

# For Q4 GGUF (recommended):
MODEL_VARIANT=q4 bash setup.sh

# For FP8 (requires 5+ A100s):
MODEL_VARIANT=fp8 bash setup.sh

# For BF16 full precision (requires 8 A100s):
MODEL_VARIANT=bf16 bash setup.sh
```

Setup takes approximately:
- Package installation: ~5 minutes
- Q4 model download (~335 GB at 500 MB/s): ~11 minutes
- FP8 model download (~600 GB): ~20 minutes

### Step 5: Run Extraction

```bash
cd /workspace/verantyx_v6

# Q4 mode:
python knowledge/runpod_deployment/run_extraction.py \
    --model q4 \
    --model-path /workspace/models/deepseek-v3-q4/ \
    --probes 15 \
    --top-experts 20 \
    --output-dir /workspace/output

# FP8 mode:
python knowledge/runpod_deployment/run_extraction.py \
    --model fp8 \
    --model-path /workspace/models/deepseek-v3-fp8/ \
    --probes 15 \
    --top-experts 20

# Monitor progress in another terminal:
tail -f /workspace/verantyx_v6/extraction.log  # (if you add tee)
```

### Step 6: Download Results

```bash
# From your local machine:
scp root@<pod_ip>:/workspace/output/pieces_600b_extracted.jsonl \
    /Users/motonishikoudai/.openclaw/workspace/verantyx_v6/pieces/

# Or use the RunPod file browser
```

### Step 7: Integrate into Verantyx

```bash
cd /Users/motonishikoudai/.openclaw/workspace/verantyx_v6

# Merge extracted pieces into main piece_db:
python -c "
from knowledge.piece_converter import PieceConverter
converter = PieceConverter()
converter.merge_into_existing(
    new_pieces=list(__import__('json').loads(l) for l in open('pieces/pieces_600b_extracted.jsonl')),
    existing_path='pieces/piece_db.jsonl',
    output_path='pieces/piece_db.jsonl',
)
"
```

---

## Configuration Reference

### `run_extraction.py` Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--stub` | False | Run in stub mode (no model needed) |
| `--model` | None | Model variant: `q4`, `fp8`, or `bf16` |
| `--model-path` | None | Path to model weights directory |
| `--domains` | all 8 | Space-separated list of domains to analyze |
| `--probes` | 15 | Probe queries per domain |
| `--top-experts` | 20 | Top experts to focus semantic extraction on |
| `--output-dir` | `pieces/` | Output directory |
| `--output-file` | `pieces_600b_extracted.jsonl` | Output filename |
| `--min-confidence` | 0.75 | Minimum confidence threshold |
| `--no-cache` | False | Skip loading routing cache |

### Environment Variables (`setup.sh`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_VARIANT` | `q4` | Model to download: `q4`, `fp8`, `bf16` |
| `PYTHON` | `python3` | Python interpreter path |

---

## Troubleshooting

### "CUDA out of memory"
- Reduce `--top-experts` to 10
- Use Q4 instead of FP8
- Add more GPU nodes or reduce batch size in the extraction code

### "Model download fails"
```bash
# Manual download:
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('bartowski/DeepSeek-V3-GGUF', 
                'DeepSeek-V3-Q4_K_M-00001-of-00034.gguf',
                local_dir='/workspace/models/deepseek-v3-q4/')
"
```

### "Not enough disk space"
- The Q4 model is 335 GB — ensure you have 400+ GB free
- You can delete the model after extraction: `rm -rf /workspace/models/`

### "vLLM install fails"
```bash
# Try specific version:
pip install vllm==0.3.3
# Or use transformers backend instead:
python run_extraction.py --model fp8 --model-path ... 
# (will auto-fall-back to transformers if vLLM unavailable)
```

### "Only 0 pieces extracted"
- Check that `--model-path` points to the correct directory
- Try `--stub` first to verify the pipeline works
- Check routing analysis: `cat knowledge/runpod_deployment/routing_cache.json`

---

## Pipeline Architecture

```
run_extraction.py
    │
    ├── Step 1: ExpertRouterAnalyzer
    │   ├── Run 120 domain probe queries through model
    │   ├── Capture expert activation per token via forward hooks
    │   └── Output: routing_cache.json
    │       {domain → [(layer, expert_id, frequency), ...]}
    │
    ├── Step 2: Top Expert Selection
    │   └── Top 20 experts per domain by activation frequency
    │
    ├── Step 3: SemanticExtractor
    │   ├── Run knowledge probes targeting top experts
    │   ├── "The derivative of x^n is ___" → "nx^(n-1)" (conf=0.95)
    │   └── Output: [ExtractedKnowledgePiece, ...]
    │
    └── Step 4: PieceConverter
        ├── Convert to Verantyx piece_db format
        ├── Validate schema
        └── Write pieces_600b_extracted.jsonl
```

---

## Domain Coverage

The pipeline extracts knowledge from 8 domains:

| Domain | Example Knowledge | Probe Count |
|--------|------------------|-------------|
| math | Power rule, quadratic formula, Euler's theorem | 15 |
| physics | Newton's laws, kinetic energy, wave equations | 15 |
| chemistry | Ideal gas law, pH, Arrhenius equation | 15 |
| biology | Central dogma, mitosis, Hardy-Weinberg | 15 |
| computer_science | Big-O, Turing machines, graph algorithms | 15 |
| history | Dates, events, causes of major events | 15 |
| literature | Authors, themes, literary devices | 15 |
| philosophy | Cogito, categorical imperative, trolley problem | 15 |

---

## Expected Output Format

Each line in `pieces_600b_extracted.jsonl` is a valid Verantyx piece:

```json
{
  "piece_id": "600b_calc_power_rule",
  "name": "Calc Power Rule",
  "description": "The derivative of x^n is nx^(n-1)",
  "in": {
    "requires": ["domain:math", "subdomain:math_calculus"],
    "slots": ["query"]
  },
  "out": {
    "produces": ["knowledge", "formula"],
    "schema": "knowledge"
  },
  "executor": "executors.knowledge.lookup",
  "confidence": 0.962,
  "tags": ["math", "math_calculus", "formula", "600b_extracted"],
  "source": "600b_weight_extraction",
  "knowledge": {
    "formula": "nx^(n-1)",
    "domain": "math",
    "subdomain": "math_calculus",
    "source_probe": "The derivative of x^n is ___",
    "expert_layer": 42,
    "expert_id": 187
  }
}
```

---

## Cost Estimate

See [cost_estimate.md](cost_estimate.md) for detailed breakdown.

**TL;DR**: ~$85 for a full extraction run on 5×A100 80GB.
Free stub mode available for testing without any GPU costs.

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `setup.sh` | One-time RunPod environment setup |
| `run_extraction.py` | Main extraction pipeline |
| `cost_estimate.md` | Detailed cost breakdown |
| `README.md` | This file |
| `routing_cache.json` | Auto-generated routing analysis cache |
