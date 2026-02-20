#!/bin/bash
# commit_score.sh — スコア改善時にコミット・プッシュ
# 使い方: ./commit_score.sh 4.20 "コミットメッセージ"

SCORE="${1:-?%}"
MSG="${2:-score improvement}"
FULL_MSG="score: ${SCORE}% bias-free — ${MSG}"

cd "$(dirname "$0")"

git add -A
if git diff --cached --quiet; then
    echo "変更なし。スキップ。"
    exit 0
fi

git commit -m "$FULL_MSG"
git push origin main 2>&1 | tail -5

echo ""
echo "✅ コミット: $FULL_MSG"
echo "🔗 https://github.com/Ag3497120/verantyx-v6"
