"""
arc/cross_life.py — Crossの人生（経験ベース記憶エンジン）

=== 設計思想 ===
断片記憶 = ラベル（何があるか知ってるだけ）
人生経験 = 実際に問題を見て、考えて、試して、失敗/成功した記録

kofdaiが113問の未解決問題をどう解くかを「人生」として記録し、
その経験パターンから新しい問題を解く。

=== 人生の構造 ===
1. 第一印象（何に見えるか）
2. 分解（どうパーツに分けるか）
3. 仮説（どんな変換だと思うか）
4. 試行（実際に試す）
5. 結果（成功/失敗、なぜ失敗したか）
6. 教訓（次に似た問題を見たら何をするか）
"""

import json
import numpy as np
from typing import List, Dict, Optional
from collections import Counter
from pathlib import Path
from dataclasses import dataclass, field, asdict
from scipy.ndimage import label as scipy_label


@dataclass
class Experience:
    task_id: str
    first_impression: List[str]
    features: Dict[str, float]
    hypothesis: str
    attempts: List[Dict]
    lesson: str
    solution_method: Optional[str] = None
    kofdai_notes: str = ""


@dataclass
class CrossLife:
    experiences: List[Experience] = field(default_factory=list)
    lesson_index: Dict[str, List[str]] = field(default_factory=dict)
    
    LIFE_PATH = Path(__file__).parent.parent / 'cross_life.json'
    
    def save(self):
        data = {
            'experiences': [asdict(e) for e in self.experiences],
            'lesson_index': self.lesson_index,
        }
        with open(self.LIFE_PATH, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls) -> 'CrossLife':
        life = cls()
        if cls.LIFE_PATH.exists():
            with open(cls.LIFE_PATH) as f:
                data = json.load(f)
            life.experiences = [Experience(**e) for e in data.get('experiences', [])]
            life.lesson_index = data.get('lesson_index', {})
        return life
    
    def add_experience(self, exp: Experience):
        self.experiences.append(exp)
        for tag in exp.first_impression:
            if tag not in self.lesson_index:
                self.lesson_index[tag] = []
            self.lesson_index[tag].append(exp.task_id)
        self.save()
    
    def recall(self, features: Dict[str, float], impressions: List[str], 
               top_k: int = 5) -> List[Experience]:
        if not self.experiences:
            return []
        
        scores = []
        for exp in self.experiences:
            imp_score = len(set(impressions) & set(exp.first_impression)) / max(len(impressions), 1)
            feat_score = 0
            common_keys = set(features.keys()) & set(exp.features.keys())
            if common_keys:
                diffs = [abs(features[k] - exp.features[k]) for k in common_keys]
                feat_score = 1.0 / (1.0 + sum(diffs))
            scores.append((imp_score * 0.6 + feat_score * 0.4, exp))
        
        scores.sort(key=lambda x: -x[0])
        return [exp for _, exp in scores[:top_k]]


def extract_features(task: dict) -> Dict[str, float]:
    train = task['train']
    gi = np.array(train[0]['input'])
    go = np.array(train[0]['output'])
    h, w = gi.shape
    oh, ow = go.shape
    bg = int(Counter(gi.flatten()).most_common(1)[0][0])
    
    features = {}
    features['input_h'] = h
    features['input_w'] = w
    features['output_h'] = oh
    features['output_w'] = ow
    features['size_ratio'] = (oh * ow) / max(h * w, 1)
    features['same_size'] = float(h == oh and w == ow)
    features['square_input'] = float(h == w)
    features['square_output'] = float(oh == ow)
    
    in_colors = set(int(v) for v in gi.flatten())
    out_colors = set(int(v) for v in go.flatten())
    features['n_input_colors'] = len(in_colors)
    features['n_output_colors'] = len(out_colors)
    features['new_colors'] = len(out_colors - in_colors)
    features['lost_colors'] = len(in_colors - out_colors)
    
    mask = (gi != bg).astype(int)
    _, n_obj = scipy_label(mask)
    features['n_objects'] = n_obj
    
    features['h_symmetric'] = float(np.array_equal(gi, gi[:, ::-1]))
    features['v_symmetric'] = float(np.array_equal(gi, gi[::-1, :]))
    
    features['has_h_separator'] = float(any(
        len(set(int(v) for v in gi[r])) == 1 and gi[r, 0] != bg
        for r in range(h)))
    features['has_v_separator'] = float(any(
        len(set(int(v) for v in gi[:, c])) == 1 and gi[0, c] != bg
        for c in range(w)))
    
    features['fg_density'] = float(mask.sum()) / (h * w)
    features['n_train'] = len(train)
    
    if h == oh and w == ow:
        features['change_ratio'] = float((gi != go).mean())
    else:
        features['change_ratio'] = 1.0
    
    return features


def extract_impressions(task: dict) -> List[str]:
    train = task['train']
    impressions = []
    
    gi = np.array(train[0]['input'])
    go = np.array(train[0]['output'])
    h, w = gi.shape
    oh, ow = go.shape
    bg = int(Counter(gi.flatten()).most_common(1)[0][0])
    
    if h == oh and w == ow:
        impressions.append('same_size')
    elif oh < h or ow < w:
        impressions.append('shrink')
    elif oh > h or ow > w:
        impressions.append('grow')
    
    if oh * ow == 1:
        impressions.append('output_single_cell')
    
    mask = (gi != bg).astype(int)
    labeled, n = scipy_label(mask)
    if n == 0:
        impressions.append('no_objects')
    elif n == 1:
        impressions.append('single_object')
    elif n == 2:
        impressions.append('two_objects')
    elif n <= 5:
        impressions.append('few_objects')
    else:
        impressions.append('many_objects')
    
    for r in range(h):
        if len(set(int(v) for v in gi[r])) == 1 and gi[r, 0] != bg:
            impressions.append('has_separator')
            break
    for c in range(w):
        if len(set(int(v) for v in gi[:, c])) == 1 and gi[0, c] != bg:
            if 'has_separator' not in impressions:
                impressions.append('has_separator')
            break
    
    if np.array_equal(gi, gi[:, ::-1]):
        impressions.append('h_symmetric')
    if np.array_equal(gi, gi[::-1, :]):
        impressions.append('v_symmetric')
    
    n_colors = len(set(int(v) for v in gi.flatten()))
    if n_colors <= 2:
        impressions.append('binary')
    elif n_colors <= 4:
        impressions.append('few_colors')
    else:
        impressions.append('many_colors')
    
    in_colors = set(int(v) for v in gi.flatten())
    out_colors = set(int(v) for v in go.flatten())
    if out_colors - in_colors:
        impressions.append('new_colors_appear')
    if in_colors - out_colors:
        impressions.append('colors_disappear')
    
    if n >= 2:
        objs = []
        for i in range(1, n+1):
            cells = list(zip(*np.where(labeled == i)))
            r0 = min(r for r,c in cells)
            c0 = min(c for r,c in cells)
            shape = frozenset((r-r0, c-c0) for r,c in cells)
            objs.append(shape)
        if len(set(objs)) == 1:
            impressions.append('repeated_shape')
    
    if h == oh and w == ow:
        diff_mask = (gi != go)
        if diff_mask.any():
            ratio = diff_mask.mean()
            if ratio < 0.1:
                impressions.append('small_change')
            elif ratio > 0.5:
                impressions.append('large_change')
            
            changed_in_bg = diff_mask & (gi == bg)
            changed_in_obj = diff_mask & (gi != bg)
            if changed_in_bg.sum() > 0 and changed_in_obj.sum() == 0:
                impressions.append('fill_bg_only')
            elif changed_in_obj.sum() > 0 and changed_in_bg.sum() == 0:
                impressions.append('recolor_obj_only')
    
    return impressions


def auto_experience(task_id: str, task: dict) -> Experience:
    from arc.kofdai_full_memory import ALL_SOLVERS
    from arc.kofdai_extra_memory import EXTRA_SOLVERS
    from arc.grid import grid_eq
    
    features = extract_features(task)
    impressions = extract_impressions(task)
    
    train_pairs = [(e['input'], e['output']) for e in task['train']]
    test_input = task['test'][0]['input']
    test_output = task['test'][0].get('output')
    
    attempts = []
    solution_method = None
    
    all_s = ALL_SOLVERS + EXTRA_SOLVERS
    for name, solver in all_s:
        try:
            ok = True
            for inp, out in train_pairs:
                pred = solver(train_pairs, inp)
                if pred is None or not grid_eq(pred, out):
                    ok = False; break
            
            if ok:
                result = solver(train_pairs, test_input)
                if result and test_output and grid_eq(result, test_output):
                    attempts.append({'method': name, 'success': True, 'error': None})
                    solution_method = name
                else:
                    attempts.append({'method': name, 'success': False, 'error': 'train_pass_test_fail'})
            else:
                attempts.append({'method': name, 'success': False, 'error': 'train_fail'})
        except Exception as e:
            attempts.append({'method': name, 'success': False, 'error': str(e)[:100]})
    
    if solution_method:
        lesson = f"Solved by {solution_method}. Key: {', '.join(impressions[:5])}"
    else:
        train_pass = [a for a in attempts if a.get('error') == 'train_pass_test_fail']
        if train_pass:
            lesson = f"Overfit: {', '.join(a['method'] for a in train_pass)}. Features: {', '.join(impressions[:5])}"
        else:
            lesson = f"No solver passed. Need new op. Features: {', '.join(impressions[:5])}"
    
    if 'has_separator' in impressions:
        hypothesis = "Panel operation (separator divides grid)"
    elif 'shrink' in impressions:
        hypothesis = "Extract/crop region or object"
    elif 'grow' in impressions:
        hypothesis = "Tile/scale/expand pattern"
    elif 'fill_bg_only' in impressions:
        hypothesis = "Fill background (enclosed, between, pattern)"
    elif 'recolor_obj_only' in impressions:
        hypothesis = "Recolor objects by property"
    elif 'many_objects' in impressions and 'same_size' in impressions:
        hypothesis = "Per-object transformation"
    elif 'repeated_shape' in impressions:
        hypothesis = "Same-shape objects: classify/sort/arrange"
    else:
        hypothesis = f"Unknown. {', '.join(impressions)}"
    
    return Experience(
        task_id=task_id, first_impression=impressions,
        features=features, hypothesis=hypothesis,
        attempts=attempts, lesson=lesson,
        solution_method=solution_method,
    )


def live_life(task_ids=None):
    life = CrossLife.load()
    existing_ids = {e.task_id for e in life.experiences}
    
    data_dir = Path('/tmp/arc-agi-2/data/training')
    if task_ids is None:
        task_ids = sorted(f.stem for f in data_dir.glob('*.json'))
    
    new_count = solved_count = 0
    for tid in task_ids:
        if tid in existing_ids:
            continue
        tf = data_dir / f'{tid}.json'
        if not tf.exists():
            continue
        with open(tf) as f:
            task = json.load(f)
        
        exp = auto_experience(tid, task)
        life.add_experience(exp)
        new_count += 1
        
        if exp.solution_method:
            solved_count += 1
            print(f'  ✓ {tid} [{exp.solution_method}]')
        else:
            print(f'  ✗ {tid} → {exp.hypothesis[:60]}')
    
    print(f'\n新規経験: {new_count}, うち解決: {solved_count}')
    print(f'総経験数: {len(life.experiences)}')
    
    unsolved = [e for e in life.experiences if not e.solution_method]
    hyp_counts = Counter(e.hypothesis.split('.')[0] for e in unsolved)
    print(f'\n未解決の仮説分布:')
    for hyp, count in hyp_counts.most_common(10):
        print(f'  {count:3d} — {hyp}')
    
    return life


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: --live | --unsolved | --stats | --recall TASK_ID")
    elif sys.argv[1] == '--live':
        live_life()
    elif sys.argv[1] == '--unsolved':
        with open('unsolved_113.json') as f:
            ids = json.load(f)
        live_life(ids)
    elif sys.argv[1] == '--stats':
        life = CrossLife.load()
        print(f'総経験数: {len(life.experiences)}')
        imp_counts = Counter()
        for e in life.experiences:
            for imp in e.first_impression:
                imp_counts[imp] += 1
        print('\n印象の分布:')
        for imp, count in imp_counts.most_common(20):
            print(f'  {count:3d} — {imp}')
