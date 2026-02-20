# RunPod Cost Estimate: DeepSeek V3 600B Knowledge Extraction

## Summary

| Phase | Duration | GPU Config | Hourly Cost | Phase Cost |
|-------|----------|-----------|-------------|------------|
| Environment setup | ~30 min | 5× A100 80GB | $13.60/hr | ~$7 |
| Model load (Q4) | ~30 min | 5× A100 80GB | $13.60/hr | ~$7 |
| Expert routing analysis | ~2 hrs | 5× A100 80GB | $13.60/hr | ~$27 |
| Semantic extraction | ~3 hrs | 5× A100 80GB | $13.60/hr | ~$41 |
| Output & cleanup | ~15 min | 5× A100 80GB | $13.60/hr | ~$3 |
| **TOTAL** | **~6.25 hrs** | — | — | **~$85** |

**Recommended budget: $75–$110**

---

## GPU Configuration Options

### Option A: 5× A100 80GB SXM (RECOMMENDED)
- **Cost**: $13.60/hr (RunPod community cloud as of 2025-02)
- **VRAM**: 5 × 80 GB = 400 GB total
- **Model format**: Q4_K_M GGUF (335 GB) or FP8 (fits with some headroom)
- **Throughput**: ~800 tokens/sec for Q4
- **Pros**: Fast routing analysis, sufficient VRAM, good cost ratio
- **Cons**: Requires multi-GPU coordination

### Option B: 3× A100 80GB SXM
- **Cost**: $8.16/hr
- **VRAM**: 3 × 80 GB = 240 GB
- **Model format**: Q4_K_M only (335 GB via CPU offloading some layers)
- **Throughput**: ~400 tokens/sec
- **Total cost**: ~$55–$75
- **Cons**: Slower, CPU offload adds latency

### Option C: 8× A100 80GB SXM (FASTEST)
- **Cost**: $21.76/hr
- **VRAM**: 8 × 80 GB = 640 GB
- **Model format**: FP8 or BF16 (native precision)
- **Throughput**: ~2000 tokens/sec
- **Total time**: ~2–3 hrs
- **Total cost**: ~$55–$65
- **Pros**: Fastest extraction, highest quality (no quantization artifacts)
- **Cons**: Higher hourly rate, may not be available

### Option D: H100 80GB NVLink × 4 (PREMIUM)
- **Cost**: ~$25/hr
- **VRAM**: 4 × 80 GB = 320 GB
- **Throughput**: ~3000 tokens/sec
- **Total time**: ~1.5–2 hrs
- **Total cost**: ~$40–$55
- **Pros**: Fastest per-dollar due to speed
- **Cons**: Higher hourly rate

---

## Detailed Phase Breakdown

### Phase 1: Environment Setup (~30 min)
```
- Install pip packages (torch, vllm, transformers): ~10 min
- Download Q4 model from HuggingFace (~335 GB at 500 MB/s): ~11 min
- GPU memory setup and model loading: ~9 min
```
**Cost: ~$7**

### Phase 2: Expert Routing Analysis (~2 hrs)
```
8 domains × 15 probes = 120 probe queries
Each probe: ~20 tokens input, ~50 tokens output = 70 tokens
Total token processing: 120 × 70 = 8,400 tokens

Plus routing hook overhead per token across 61 layers = ~511,400 expert evaluations
At 800 tok/sec (Q4): 8,400 / 800 ≈ 10.5 sec of pure inference
Routing hook overhead × ~10: ~105 sec per query
Total: 120 queries × 105 sec ≈ 3.5 hrs

(Optimized with batching and hook sampling: ~2 hrs realistic)
```
**Cost: ~$27**

### Phase 3: Semantic Knowledge Extraction (~3 hrs)
```
8 domains × 20 top experts × 15 probes per expert = 2,400 probe completions
Each completion: ~100 tokens
Total: 240,000 tokens at 800 tok/sec = 300 sec = 5 min of pure inference

Overhead (expert activation tracking, hook processing): ~20×
Total: ~100 min per pass; with 3 passes for quality: ~5 hrs
(Optimized with parallel expert probing: ~3 hrs realistic)
```
**Cost: ~$41**

### Phase 4: Output & Cleanup (~15 min)
```
- Write pieces_600b_extracted.jsonl
- Compress and download outputs
- Instance teardown
```
**Cost: ~$3**

---

## Expected Output

| Metric | Estimate |
|--------|----------|
| Total pieces extracted | 150–300 |
| High-confidence pieces (>0.85) | 100–200 |
| Unique domains covered | 8 |
| Unique expert neurons mapped | 160 (20 per domain) |
| Output file size | ~2–5 MB |
| Stub mode output (free) | 100+ pieces |

---

## Cost Optimization Tips

1. **Use spot instances**: RunPod offers 20–40% discounts on interruptible pods
   - Risk: Instance may be preempted mid-extraction
   - Mitigation: Checkpoint routing analysis to JSON after Step 1

2. **Start with Q4 model**: 
   - Q4_K_M gives ~95% of BF16 quality at 1/3 the VRAM cost
   - Sufficient for knowledge extraction (you're extracting facts, not doing RLHF)

3. **Cache routing results**: 
   - The `routing_cache.json` saves Step 2 output
   - If Step 3 fails, you can restart from cache at no additional Step 2 cost

4. **Use smaller probes_per_domain for budget runs**:
   - `--probes 5` instead of `--probes 15` for ~3× cost reduction
   - Routing analysis quality decreases but still usable

5. **Run stub mode first (FREE)**:
   - `python run_extraction.py --stub`
   - Produces 100+ realistic pieces from hardcoded knowledge
   - Good enough to test the full Verantyx pipeline before spending on GPU time

---

## Cloud Provider Comparison (5×A100 equivalent)

| Provider | Config | Hourly | Notes |
|----------|--------|--------|-------|
| RunPod Community | 5×A100 80GB | ~$13.60 | Best price, variable availability |
| RunPod Secure | 5×A100 80GB | ~$16.50 | More reliable |
| Lambda Labs | 8×A100 80GB | ~$15.00 | Fixed config only |
| Vast.ai | 5×A100 80GB | ~$11–15 | Cheapest, less reliable |
| AWS p4d.24xlarge | 8×A100 40GB | ~$32.80 | Most reliable, most expensive |
| GCP a2-ultragpu-8g | 8×A100 80GB | ~$30.00 | Premium |

**Recommendation**: RunPod Community for cost, Lambda Labs for reliability.

---

## Checkpointing Strategy

To avoid losing work if the instance is preempted:

```python
# The pipeline automatically caches after Step 2:
# knowledge/runpod_deployment/routing_cache.json

# Resume from cache:
python run_extraction.py --model q4 \
    --model-path /workspace/models/deepseek-v3-q4/ \
    # (cache is auto-loaded if routing_cache.json exists)

# Skip routing (cache only), re-run semantic extraction:
python run_extraction.py --stub  # Uses cached route data
```

---

*Estimates based on RunPod pricing as of February 2025 and benchmark throughput numbers for DeepSeek V3 Q4_K_M on A100 SXM. Actual costs may vary ±20% based on instance availability and network speed.*
