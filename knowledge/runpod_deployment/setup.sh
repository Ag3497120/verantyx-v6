#!/bin/bash
# =============================================================================
# Verantyx 600B Knowledge Extraction - RunPod Setup Script
# =============================================================================
# Run this ONCE after launching your RunPod instance before running
# run_extraction.py.
#
# Recommended pod config:
#   GPU: 8× A100 80GB (640GB VRAM total) ← CURRENT CONFIG
#   RAM: 256GB+ system RAM
#   Storage: 800GB NVMe SSD (FP8 model ~670GB + outputs)
#   Image: PyTorch 2.1+ / CUDA 12.1+
# =============================================================================

set -euo pipefail

echo "========================================="
echo "  Verantyx 600B Extraction Setup"
echo "========================================="
date

# ── Config ────────────────────────────────────
WORK_DIR="/workspace/verantyx_v6"
MODEL_DIR="/workspace/models"
OUTPUT_DIR="/workspace/output"
PYTHON=${PYTHON:-python3}

# Model variant to use:
#   "q4"     → Q4_K_M GGUF (335GB, via llama.cpp)  - cheapest
#   "fp8"    → FP8 safetensors (670GB, via vLLM)    - recommended for 8xA100
#   "bf16"   → BF16 safetensors (1.34TB, 8xA100)   - most accurate (needs >1TB VRAM)
MODEL_VARIANT=${MODEL_VARIANT:-"fp8"}

# ── System Checks ──────────────────────────────
echo ""
echo "[SETUP] Checking system..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  WARNING: nvidia-smi not found"
echo "  Python: $($PYTHON --version)"
echo "  CPU cores: $(nproc)"
echo "  RAM: $(free -h | awk '/^Mem:/{print $2}')"
df -h /workspace | tail -1 | awk '{print "  Disk: " $4 " free"}'

# ── Create Directories ─────────────────────────
echo ""
echo "[SETUP] Creating directories..."
mkdir -p "$WORK_DIR" "$MODEL_DIR" "$OUTPUT_DIR"

# ── Install Core Dependencies ──────────────────
echo ""
echo "[SETUP] Installing Python dependencies..."
$PYTHON -m pip install --upgrade pip --quiet

# Core scientific stack
$PYTHON -m pip install \
    numpy scipy \
    torch torchvision \
    --quiet

# Transformers ecosystem
$PYTHON -m pip install \
    transformers==4.40.0 \
    accelerate==0.28.0 \
    safetensors==0.4.3 \
    tokenizers \
    sentencepiece \
    --quiet

# vLLM for FP8/BF16 serving (large install - ~5min)
if [ "$MODEL_VARIANT" != "q4" ]; then
    echo "[SETUP] Installing vLLM (this takes a few minutes)..."
    $PYTHON -m pip install vllm==0.4.0 --quiet
fi

# llama.cpp Python bindings for Q4
if [ "$MODEL_VARIANT" = "q4" ]; then
    echo "[SETUP] Installing llama-cpp-python..."
    CMAKE_ARGS="-DLLAMA_CUDA=ON" \
    $PYTHON -m pip install llama-cpp-python \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 \
        --quiet 2>/dev/null || \
    $PYTHON -m pip install llama-cpp-python --quiet
fi

# Output/utilities
$PYTHON -m pip install \
    tqdm \
    rich \
    huggingface_hub \
    --quiet

echo "[SETUP] Dependencies installed ✓"

# ── Download Model ─────────────────────────────
echo ""
echo "[SETUP] Downloading DeepSeek V3 ($MODEL_VARIANT)..."

case "$MODEL_VARIANT" in
    "q4")
        echo "  Downloading Q4_K_M GGUF (~335GB)..."
        echo "  Source: Hugging Face - bartowski/DeepSeek-V3-GGUF"
        mkdir -p "$MODEL_DIR/deepseek-v3-q4"
        
        # Download in parts (the GGUF is split into ~34 files of ~10GB each)
        $PYTHON -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='bartowski/DeepSeek-V3-GGUF',
    local_dir='$MODEL_DIR/deepseek-v3-q4',
    allow_patterns=['*Q4_K_M*'],
    resume_download=True,
)
print('Downloaded!')
" || echo "  WARNING: HF download failed. Try manual download below."
        
        echo ""
        echo "  Alternative manual download:"
        echo "  wget 'https://huggingface.co/bartowski/DeepSeek-V3-GGUF/resolve/main/DeepSeek-V3-Q4_K_M-00001-of-00034.gguf' -P $MODEL_DIR/deepseek-v3-q4/"
        ;;
        
    "fp8")
        echo "  Downloading FP8 (~600GB via HuggingFace)..."
        echo "  Source: deepseek-ai/DeepSeek-V3 (FP8 branch)"
        mkdir -p "$MODEL_DIR/deepseek-v3-fp8"
        
        $PYTHON -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='deepseek-ai/DeepSeek-V3',
    local_dir='$MODEL_DIR/deepseek-v3-fp8',
    resume_download=True,
    ignore_patterns=['*.md', '*.txt'],
)
" || echo "  WARNING: Download failed. Manual: git lfs clone https://huggingface.co/deepseek-ai/DeepSeek-V3 $MODEL_DIR/deepseek-v3-fp8"
        ;;
        
    "bf16")
        echo "  BF16 requires 8×A100 80GB. Download via:"
        echo "  git lfs clone https://huggingface.co/deepseek-ai/DeepSeek-V3 $MODEL_DIR/deepseek-v3-bf16"
        ;;
esac

# ── Upload Project Code ────────────────────────
echo ""
echo "[SETUP] Setting up project code..."
echo "  Upload your local project to RunPod:"
echo "  rsync -avz --exclude='venv/' --exclude='*.log' --exclude='*.json' \\"
echo "    /Users/motonishikoudai/.openclaw/workspace/verantyx_v6/ \\"
echo "    root@<pod_ip>:/workspace/verantyx_v6/"
echo ""
echo "  OR use RunPod's file upload UI / S3 sync"

# ── Verify Setup ───────────────────────────────
echo ""
echo "[SETUP] Verifying installation..."
$PYTHON -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'  GPU {i}: {name} ({mem:.0f}GB)')
"

echo ""
echo "========================================="
echo "  Setup Complete!"
echo "  Run: python run_extraction.py --stub"
echo "  Or:  python run_extraction.py --model $MODEL_VARIANT"
echo "========================================="
