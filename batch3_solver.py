import json
import os
import sys
import copy

DATA_DIR = "/private/tmp/arc-agi-2/data/training"
OUT_DIR = os.path.expanduser("~/verantyx_v6/synth_results")

missing = ["b94a9452", "baf41dbf", "bb52a14b", "bc93ec48", "bd14c3bf", "bd283c4a", "bd5af378", "beb8660c", "c3202e5a", "c35c1b4c", "c3fa4749", "c444b776", "c4d1a9ae", "c6141b15", "c61be7dc", "c62e2108", "c64f1187", "c658a4bd", "c6e1b8da", "c803e39c", "c87289bb", "c8b7cc0f", "c8cbb738", "c920a713", "c92b942c", "c97c0139", "ca8de6ea", "cbded52d", "cc9053aa", "cdecee7f", "ce039d91", "ce602527", "cf133acc", "cf98881b", "cfb2ce5a", "d017b73f", "d06dbe63", "d07ae81c", "d22278a0", "d23f8c26", "d255d7a7", "d2abd087", "d2acf2cb", "d304284e", "d37a1ef5", "d406998b", "d43fd935", "d4469b4b", "d47aa2ff", "d492a647", "d4c90558", "d4f3cd78", "d56f2372", "d5c634a2", "d5d6de2d", "d6542281", "d687bc17", "d6ad076f", "d6e50e54", "d749d46f", "d753a70b", "d89b689b", "d8c310e9", "d90796e8", "d931c21c", "d93c6891", "d94c3b52", "d968ffd4", "d9f24cd1", "da2b0fe3", "da6e95e5", "db118e2a"]

def load_task(tid):
    path = os.path.join(DATA_DIR, f"{tid}.json")
    return json.load(open(path))

def test_transform(fn, examples):
    for ex in examples:
        out = fn(copy.deepcopy(ex['input']))
        if out != ex['output']:
            return False
    return True

def save(tid, code):
    with open(os.path.join(OUT_DIR, f"{tid}.py"), 'w') as f:
        f.write(code)

# Show all tasks data briefly
for tid in missing:
    task = load_task(tid)
    train = task['train']
    ex0 = train[0]
    print(f"\n=== {tid} ===")
    print(f"  Examples: {len(train)}")
    print(f"  Input shape: {len(ex0['input'])}x{len(ex0['input'][0])}")
    print(f"  Output shape: {len(ex0['output'])}x{len(ex0['output'][0])}")
    # print first example
    for row in ex0['input']:
        print(f"  IN:  {row}")
    for row in ex0['output']:
        print(f"  OUT: {row}")
    if len(train) > 1:
        ex1 = train[1]
        for row in ex1['input']:
            print(f"  IN2: {row}")
        for row in ex1['output']:
            print(f"  OUT2: {row}")

