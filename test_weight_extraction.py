"""
Weight-Based Knowledge Extraction Test

重みファイルベースの非発火知識抽出テスト
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from knowledge.weight_loader import WeightLoaderStub
from knowledge.expert_profiler import ExpertProfiler
from knowledge.cross_mapper import CrossStructureMapper
from knowledge.weight_extractor import WeightKnowledgeExtractor
from core.ir import Domain


def test_basic_pipeline():
    """基本パイプラインのテスト（スタブモード）"""
    print("="*80)
    print("Weight-Based Knowledge Extraction Test (Stub Mode)")
    print("="*80)
    print()
    
    # Step 1: WeightLoader初期化（スタブモード）
    print("Step 1: Initializing WeightLoader (stub mode)...")
    loader = WeightLoaderStub()
    print(f"  - Model: Stub (no actual model file)")
    print(f"  - Layers: {loader.num_layers}")
    print(f"  - Experts per layer: {loader.num_experts}")
    print()
    
    # メモリ使用量推定
    memory_est = loader.estimate_memory_usage()
    print(f"  - Memory per expert: {memory_est['per_expert_mb']:.1f} MB")
    print(f"  - Total (all experts): {memory_est['all_experts_gb']:.1f} GB")
    print()
    
    # Step 2: Expert重みのロードテスト
    print("Step 2: Loading expert weights...")
    W = loader.load_expert_weights(layer=30, expert_id=42, component="gate_proj")
    print(f"  - Weight shape: {W.shape}")
    print(f"  - Weight mean: {W.mean():.6f}")
    print(f"  - Weight std: {W.std():.6f}")
    print()
    
    # Step 3: ExpertProfiler初期化
    print("Step 3: Profiling experts...")
    profiler = ExpertProfiler(loader)
    
    # サンプルexpertをプロファイリング
    sample_experts = [
        (10, 0),   # 浅い層、最初のexpert
        (30, 42),  # 中層、中間expert
        (50, 255)  # 深い層、最後のexpert
    ]
    
    profiles = {}
    for layer, expert_id in sample_experts:
        print(f"  - Profiling L{layer}E{expert_id}...")
        scores = profiler.profile_expert(layer, expert_id)
        profiles[(layer, expert_id)] = scores
        
        # Top 3 domains
        top_domains = sorted(scores.items(), key=lambda x: -x[1])[:3]
        print(f"    Top domains: {[f'{d.value}:{s:.3f}' for d, s in top_domains]}")
    
    print()
    
    # Step 4: Cross構造にマッピング
    print("Step 4: Building Cross structure...")
    cross_mapper = CrossStructureMapper(profiler)
    cross_mapper.build_cross_structure(profiles)
    
    for (layer, expert_id), coords in cross_mapper.cross_space.items():
        x, y, z = coords
        print(f"  - L{layer}E{expert_id}: ({x:.3f}, {y:.3f}, {z:.3f})")
    
    print()
    
    # Step 5: 知識抽出テスト
    print("Step 5: Extracting knowledge from weights...")
    extractor = WeightKnowledgeExtractor(loader, cross_mapper)
    
    # テスト問題
    test_problems = [
        {
            "problem": "What is the largest order of a torsion subgroup of an elliptic curve?",
            "domain": Domain.NUMBER_THEORY
        },
        {
            "problem": "Calculate the determinant of a 3x3 matrix",
            "domain": Domain.LINEAR_ALGEBRA
        },
        {
            "problem": "Prove that the halting problem is undecidable",
            "domain": Domain.LOGIC_PROPOSITIONAL
        }
    ]
    
    for i, test in enumerate(test_problems, 1):
        print(f"\n--- Test Problem {i} ---")
        print(f"Problem: {test['problem'][:60]}...")
        print(f"Domain: {test['domain'].value}")
        
        # 知識抽出
        knowledge_pieces = extractor.extract_knowledge(
            problem=test['problem'],
            domain=test['domain'],
            k_experts=3
        )
        
        print(f"Extracted {len(knowledge_pieces)} knowledge pieces:")
        for kp in knowledge_pieces:
            print(f"  - {kp.name}")
            print(f"    Confidence: {kp.confidence:.3f}")
            print(f"    Coords: ({kp.coords[0]:.3f}, {kp.coords[1]:.3f}, {kp.coords[2]:.3f})")
            print(f"    Description: {kp.description[:80]}...")
    
    print()
    print("="*80)
    print("Test Complete")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Download DeepSeek V3.2 model files (~600GB)")
    print("2. Replace WeightLoaderStub with DeepSeekWeightLoader")
    print("3. Run full profiling on all 15,616 experts")
    print("4. Build complete Cross structure")
    print("5. Integrate with Verantyx V6 pipeline")


if __name__ == "__main__":
    test_basic_pipeline()
