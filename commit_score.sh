#!/bin/bash
# commit_score.sh — スコア改善時にコミット・プッシュ
# 使い方: ./commit_score.sh 4.20 "CEGISバグ修正 + MCQ verifier統合"

SCORE=${1:-"?%"}
MSG=${2:-"score improvement"}
FULL_MSG="score: ${SCORE}% bias-free — ${MSG}"

cd /Users/motonishikoudai/.openclaw/workspace/verantyx_v6

git add -A
git diff --cached --quiet && echo "変更なし。スキップ。" && exit 0

git commit -m "$FULL_MSG"
git push origin main 2>&1 | tail -5

echo ""
echo "✅ コミット完了: $FULL_MSG"
echo "🔗 https://github.com/Ag3497120/verantyx-v6"
