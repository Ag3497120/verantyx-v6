# Verantyx V6 - Heartbeat Status (Updated 09:40 JST 2026-02-28)

**フェーズ**: v75 eval完了 → **237/1000 (23.7%)** (v74と同スコア)

---

## 📊 本日の実装

### 新規モジュール (4ファイル)
- `arc/object_ir.py` — オブジェクトIR (connected component, CellRoleSignature, enclosed regions)
- `arc/role_nb.py` — role-aware NB learner (6戦略: compact_only, nb_canonical_plus_compact等)
- `arc/topology_solver.py` — 複数enclosure仮説 (4conn/8conn/sealed)
- `arc/object_program.py` — ObjectProgramTree (recolor_by_rank, remove_by_color等)

### nb_abstract.py 拡張
- `learn_rotation_invariant_nb_rule` — D4群回転/反射不変NB (カバレッジ26%→2%改善)
- `learn_rotsym_count_nb_rule` — 超粗NBカウントルール

### 結果
| モジュール | train上学習成功 | test正解 | 既存と重複 |
|---|---|---|---|
| role_nb | 72/560 | 3 | **3/3 重複** |
| topology_solver | 0 | 0 | — |
| object_program | 0 | 0 | — |
| rot_inv_nb | 3/148(ver=5) | 0 | — |

## 🔍 重要な知見

1. **ver=5の148タスクはNBルールでは原理的に解けない** — 145/148がinconsistent (同じ局所パターンが異なる出力に対応)
2. **残り763タスクに単純変換(color_map/rotate/flip)は存在しない** — 全て既存ソルバーで処理済み
3. **enclosed regionベースのfillは刺さらない** — ほとんどのbg regionがborder touchingする
4. **object recolor/removeルールも刺さらない** — 残りタスクは単純なオブジェクト操作を超えている

## 📋 次のアクション

- [ ] puzzle_lang DSLプリミティブ拡張 (地道だが確実な+1)
- [ ] iterative_cross の組み合わせ空間拡大
- [ ] program_search の探索深度拡張
- [ ] kofdai のアイデア待ち

## 🔧 eval起動コマンド
```bash
cd ~/verantyx_v6
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null
nohup python3 -u -m arc.eval_cross_engine --split training > arc_v76_full.log 2>&1 &
```
