"""
Minimal Weight Extraction Test (no heavy computation)
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

print('=== Minimal Weight Extraction Test ===')

# Test 1: Import check
try:
    from knowledge.weight_loader import WeightLoaderStub
    print('✅ WeightLoader imported')
except Exception as e:
    print(f'❌ WeightLoader import failed: {e}')
    sys.exit(1)

# Test 2: Create loader
try:
    loader = WeightLoaderStub()
    print(f'✅ WeightLoader initialized: {loader.num_experts} experts')
except Exception as e:
    print(f'❌ WeightLoader init failed: {e}')
    sys.exit(1)

# Test 3: Load weight (without SVD)
try:
    W = loader.load_expert_weights(layer=10, expert_id=5)
    print(f'✅ Weight loaded: shape {W.shape}')
except Exception as e:
    print(f'❌ Weight load failed: {e}')
    sys.exit(1)

print()
print('All tests passed! ✅')
print()
print('Next: Download DeepSeek V3.2 model (~600GB)')
print('  git clone https://huggingface.co/deepseek-ai/DeepSeek-V3-Base')
