"""
Verantyx 600B Knowledge Extraction - Main RunPod Pipeline

Full end-to-end extraction pipeline:
  1. Initialize model (DeepSeek V3 Q4/FP8/BF16 via vLLM or llama.cpp)
  2. Run expert routing analysis (500 domain-specific probe queries)
  3. Identify top 20 experts per domain
  4. Extract semantic knowledge from top experts via probing
  5. Convert to Verantyx piece_db.jsonl format
  6. Save as pieces_600b_extracted.jsonl

Usage:
    # Test without model (stub mode):
    python run_extraction.py --stub

    # Real Q4 extraction on RunPod:
    python run_extraction.py --model q4 --model-path /workspace/models/deepseek-v3-q4/

    # Real FP8 extraction (requires 5×A100 80GB):
    python run_extraction.py --model fp8 --model-path /workspace/models/deepseek-v3-fp8/
"""

import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any

# ── Add project root to path ────────────────
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Local imports ────────────────────────────
from knowledge.expert_router_analyzer import ExpertRouterAnalyzer, DOMAIN_PROBES
from knowledge.semantic_extractor import SemanticExtractor
from knowledge.piece_converter import PieceConverter

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

DEFAULT_CONFIG = {
    "domains": list(DOMAIN_PROBES.keys()),
    "probes_per_domain": 15,          # 15 probes × 8 domains = 120 queries
    "top_experts_per_domain": 20,     # Focus semantic extraction on top 20 experts
    "output_dir": str(PROJECT_ROOT / "pieces"),
    "output_filename": "pieces_600b_extracted.jsonl",
    "routing_cache": str(SCRIPT_DIR / "routing_cache.json"),
    "min_confidence": 0.75,
}


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────

def load_model_vllm(model_path: str, gpu_memory_util: float = 0.88):
    """Load DeepSeek V3 via vLLM (FP8 on 8×A100 80GB = 640GB VRAM).

    FP8 model is ~670GB. With 640GB VRAM + cpu_offload_gb for embeddings,
    vLLM handles this automatically.
    """
    print(f"[LOAD] Loading DeepSeek V3 FP8 via vLLM")
    print(f"[LOAD] Model path: {model_path}")
    print(f"[LOAD] GPU memory utilization: {gpu_memory_util}")

    try:
        from vllm import LLM, SamplingParams
        import torch

        num_gpus = torch.cuda.device_count()
        print(f"[LOAD] Detected {num_gpus} GPUs  (640GB VRAM total)")

        # For FP8 on 8×A100: slightly over 640GB, use cpu_offload_gb for
        # embedding layers (~15GB) to fit the rest in VRAM.
        llm = LLM(
            model=model_path,
            tensor_parallel_size=num_gpus,       # 8-way tensor parallel
            gpu_memory_utilization=gpu_memory_util,
            trust_remote_code=True,
            dtype="float8_e4m3fn",               # native FP8
            max_model_len=2048,                  # keep KV cache small
            cpu_offload_gb=20,                   # offload ~20GB to CPU for embeddings
            enforce_eager=False,                 # CUDA graphs for speed
        )
        return llm, None  # vLLM handles tokenization internally

    except ImportError:
        raise RuntimeError("vLLM not installed. Run: pip install vllm")
    except Exception as e:
        raise RuntimeError(f"Failed to load model via vLLM: {e}")


def load_model_llama_cpp(model_path: str):
    """Load DeepSeek V3 Q4 via llama.cpp."""
    print(f"[LOAD] Loading Q4 model via llama-cpp-python from {model_path}")

    # Find the GGUF file
    model_dir = Path(model_path)
    gguf_files = list(model_dir.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No GGUF files found in {model_path}")

    # Use the first part of a split GGUF
    main_gguf = sorted(gguf_files)[0]
    print(f"[LOAD] Loading: {main_gguf.name}")

    try:
        from llama_cpp import Llama

        model = Llama(
            model_path=str(main_gguf),
            n_gpu_layers=-1,  # Offload all layers to GPU
            n_ctx=4096,
            n_batch=512,
            verbose=False,
        )
        return model, None

    except ImportError:
        raise RuntimeError("llama-cpp-python not installed. Run: pip install llama-cpp-python")


def load_model_transformers(model_path: str):
    """Load DeepSeek V3 via Hugging Face Transformers (for routing hooks)."""
    print(f"[LOAD] Loading model via transformers from {model_path}")

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        print(f"[LOAD] Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
        return model, tokenizer

    except ImportError:
        raise RuntimeError("transformers not installed. Run: pip install transformers accelerate")


# ──────────────────────────────────────────────
# Extraction Pipeline
# ──────────────────────────────────────────────

class ExtractionPipeline:
    """
    Full extraction pipeline: probe → route → extract → convert → save.
    """

    def __init__(self, config: Dict[str, Any], stub: bool = False):
        self.config = config
        self.stub = stub
        self.model = None
        self.tokenizer = None
        self.start_time = time.time()

        print("=" * 60)
        print("  Verantyx 600B Knowledge Extraction Pipeline")
        print("=" * 60)
        print(f"  Mode: {'STUB (no model)' if stub else 'REAL'}")
        print(f"  Domains: {', '.join(config['domains'])}")
        print(f"  Probes per domain: {config['probes_per_domain']}")
        print(f"  Top experts per domain: {config['top_experts_per_domain']}")
        print(f"  Output: {config['output_dir']}/{config['output_filename']}")
        print()

    def run(self) -> int:
        """
        Execute the full extraction pipeline.

        Returns:
            Number of pieces extracted.
        """
        try:
            # Step 1: Expert routing analysis
            print("[PIPELINE] Step 1/4: Expert Routing Analysis")
            routing_result = self._run_routing_analysis()

            # Step 2: Identify top experts per domain
            print("\n[PIPELINE] Step 2/4: Identifying Top Experts")
            top_experts = self._identify_top_experts(routing_result)

            # Step 3: Semantic knowledge extraction
            print("\n[PIPELINE] Step 3/4: Semantic Knowledge Extraction")
            pieces = self._extract_semantic_knowledge(top_experts)

            # Step 4: Save to piece_db format
            print("\n[PIPELINE] Step 4/4: Saving to piece_db.jsonl")
            count = self._save_pieces(pieces)

            elapsed = time.time() - self.start_time
            print(f"\n{'=' * 60}")
            print(f"  Extraction Complete!")
            print(f"  Pieces extracted: {count}")
            print(f"  Elapsed time: {elapsed:.1f}s")
            print(f"  Output: {self.config['output_dir']}/{self.config['output_filename']}")
            print(f"{'=' * 60}")

            return count

        except KeyboardInterrupt:
            print("\n[PIPELINE] Interrupted by user")
            return 0
        except Exception as e:
            print(f"\n[PIPELINE] FATAL ERROR: {e}")
            traceback.print_exc()
            return 0

    def _run_routing_analysis(self):
        """Step 1: Run expert routing analysis."""
        cache_path = self.config.get("routing_cache")

        # Try loading from cache first
        if cache_path and Path(cache_path).exists() and not self.stub:
            print(f"[PIPELINE]   Loading routing cache: {cache_path}")
            try:
                return ExpertRouterAnalyzer.load_result(cache_path)
            except Exception as e:
                print(f"[PIPELINE]   Cache load failed: {e}. Running fresh analysis.")

        analyzer = ExpertRouterAnalyzer(
            model=self.model,
            tokenizer=self.tokenizer,
            stub=self.stub,
            cache_path=cache_path,
        )

        result = analyzer.analyze_all_domains(
            domains=self.config["domains"],
            probes_per_domain=self.config["probes_per_domain"],
        )

        # Print summary
        for domain in self.config["domains"]:
            experts = result.domain_experts.get(domain, [])
            print(f"[PIPELINE]   {domain}: {len(experts)} expert profiles found")

        return result

    def _identify_top_experts(self, routing_result) -> Dict[str, List]:
        """Step 2: Get top N experts per domain."""
        top_experts = {}
        n = self.config["top_experts_per_domain"]

        for domain in self.config["domains"]:
            profiles = routing_result.domain_experts.get(domain, [])
            top = profiles[:n]
            top_experts[domain] = [(p.layer, p.expert_id) for p in top]

            if top:
                best = top[0]
                print(f"[PIPELINE]   {domain}: top expert L{best.layer}E{best.expert_id} "
                      f"(freq={best.activation_frequency:.4f})")

        return top_experts

    def _extract_semantic_knowledge(self, top_experts: Dict[str, List]):
        """Step 3: Extract semantic knowledge via probing."""
        extractor = SemanticExtractor(
            model=self.model,
            tokenizer=self.tokenizer,
            stub=self.stub,
        )

        all_pieces = []

        for domain in self.config["domains"]:
            experts = top_experts.get(domain, [])
            print(f"[PIPELINE]   Probing {domain} with {len(experts)} expert constraints...")

            pieces = extractor.extract_from_domain(domain, top_experts=experts)
            all_pieces.extend(pieces)
            print(f"[PIPELINE]   → {len(pieces)} pieces from {domain}")

        return all_pieces

    def _save_pieces(self, pieces) -> int:
        """Step 4: Convert and save pieces."""
        output_dir = Path(self.config["output_dir"])
        output_path = output_dir / self.config["output_filename"]

        converter = PieceConverter()

        # Convert to piece_db format
        db_pieces = [p.to_piece_db_format() for p in pieces]

        # Filter by confidence
        min_conf = self.config.get("min_confidence", 0.75)
        before = len(db_pieces)
        db_pieces = [p for p in db_pieces if p.get("confidence", 0) >= min_conf]
        if before != len(db_pieces):
            print(f"[PIPELINE]   Filtered {before - len(db_pieces)} low-confidence pieces")

        # Validate
        valid_pieces = [p for p in db_pieces if converter.validate(p)]
        print(f"[PIPELINE]   Valid pieces: {len(valid_pieces)}/{len(db_pieces)}")

        # Write
        count = converter.write_jsonl(valid_pieces, str(output_path))

        # Print stats
        stats = converter.stats(valid_pieces)
        print(f"[PIPELINE]   Mean confidence: {stats['mean_confidence']:.3f}")
        print(f"[PIPELINE]   Domains covered: {len(stats['domains'])}")

        return count


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Verantyx 600B Knowledge Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stub mode (no model needed):
  python run_extraction.py --stub

  # Q4 mode (335GB GGUF via llama.cpp):
  python run_extraction.py --model q4 --model-path /workspace/models/deepseek-v3-q4/

  # FP8 mode (600GB via vLLM, 5×A100):
  python run_extraction.py --model fp8 --model-path /workspace/models/deepseek-v3-fp8/

  # BF16 mode (1.2TB via vLLM, 8×A100):
  python run_extraction.py --model bf16 --model-path /workspace/models/deepseek-v3-bf16/
        """
    )

    parser.add_argument("--stub", action="store_true",
                        help="Run in stub mode (no real model needed)")
    parser.add_argument("--model", choices=["q4", "fp8", "bf16"], default=None,
                        help="Model variant to use")
    parser.add_argument("--model-path", default=None,
                        help="Path to model weights directory")
    parser.add_argument("--domains", nargs="+", default=None,
                        help="Domains to analyze (default: all)")
    parser.add_argument("--probes", type=int, default=15,
                        help="Probe queries per domain (default: 15)")
    parser.add_argument("--top-experts", type=int, default=20,
                        help="Top experts per domain (default: 20)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for pieces")
    parser.add_argument("--output-file", default="pieces_600b_extracted.jsonl",
                        help="Output filename (default: pieces_600b_extracted.jsonl)")
    parser.add_argument("--min-confidence", type=float, default=0.75,
                        help="Minimum confidence threshold (default: 0.75)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip routing cache")

    return parser.parse_args()


def main():
    args = parse_args()

    # Build config
    config = dict(DEFAULT_CONFIG)

    if args.domains:
        config["domains"] = args.domains
    if args.probes:
        config["probes_per_domain"] = args.probes
    if args.top_experts:
        config["top_experts_per_domain"] = args.top_experts
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.output_file:
        config["output_filename"] = args.output_file
    if args.min_confidence:
        config["min_confidence"] = args.min_confidence
    if args.no_cache:
        config["routing_cache"] = None

    # Validate
    stub = args.stub or (args.model is None)

    if not stub and args.model_path is None:
        print("ERROR: --model-path required when --model is specified")
        print("Use --stub for testing without a real model")
        sys.exit(1)

    # Create pipeline
    pipeline = ExtractionPipeline(config=config, stub=stub)

    # Load model if not stub
    if not stub and args.model_path:
        print(f"[MAIN] Loading {args.model} model from {args.model_path}...")
        try:
            if args.model == "q4":
                pipeline.model, pipeline.tokenizer = load_model_llama_cpp(args.model_path)
            elif args.model in ("fp8", "bf16"):
                # Try transformers first (needed for routing hooks)
                try:
                    pipeline.model, pipeline.tokenizer = load_model_transformers(args.model_path)
                except Exception as e:
                    print(f"  Transformers load failed ({e}), falling back to vLLM")
                    pipeline.model, pipeline.tokenizer = load_model_vllm(args.model_path)
        except Exception as e:
            print(f"ERROR loading model: {e}")
            print("Falling back to stub mode")
            pipeline.stub = True

    # Run extraction
    count = pipeline.run()

    if count >= 100:
        print(f"\n✓ SUCCESS: Extracted {count} pieces (≥ 100 target met)")
    elif count > 0:
        print(f"\n⚠ PARTIAL: Extracted {count} pieces (< 100 target)")
    else:
        print(f"\n✗ FAILED: No pieces extracted")
        sys.exit(1)


if __name__ == "__main__":
    main()
