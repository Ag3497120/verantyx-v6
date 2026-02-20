"""Check if worlds are being generated for CEGIS"""
import json
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from cegis.worldgen import WorldGenerator

# Initialize world generator
wg = WorldGenerator()

# Test various domains
domains_to_test = [
    ("number", {}),
    ("matrix", {}),
    ("polynomial", {}),
    ("propositional", {}),
    ("graph", {}),
    ("set", {}),
]

print("="*80)
print("WORLD GENERATION TEST")
print("="*80)

for domain, params in domains_to_test:
    worlds = wg.generate(domain, params)
    print(f"\n{domain}: generated {len(worlds)} worlds")
    if worlds:
        print(f"  Sample: {worlds[0]}")
    else:
        print(f"  WARNING: NO WORLDS GENERATED!")

# Test with unknown domain
print("\n" + "="*80)
print("Testing unknown domain...")
unknown_worlds = wg.generate("unknown_domain_xyz", {})
print(f"unknown_domain_xyz: {len(unknown_worlds)} worlds")

print("\n" + "="*80)
print("DIAGNOSIS:")
if any(len(wg.generate(d, p)) == 0 for d, p in domains_to_test):
    print("  ❌ Some domains generate 0 worlds!")
    print("  This means CEGIS can't properly verify candidates")
else:
    print("  ✓ All tested domains generate worlds")

print("="*80)
