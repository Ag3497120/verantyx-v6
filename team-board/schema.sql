-- team-board の表
CREATE TABLE IF NOT EXISTS members (
  token_hash TEXT PRIMARY KEY,   -- 合言葉の SHA-256（合言葉そのものは保存しない）
  name TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'member'   -- 'leader' だけが意見を消せる
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  author TEXT,                   -- 名前なしで共有したときは NULL（誰が書いたかは残らない）
  text TEXT NOT NULL,
  created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS reactions (
  message_id INTEGER NOT NULL,
  member_hash TEXT NOT NULL,
  kind TEXT NOT NULL,            -- 'like' | 'more'
  PRIMARY KEY (message_id, member_hash, kind)
);

-- 送りすぎを防ぐための回数記録。1 時間ごとの合計だけを持ち、細かい時刻は残さない
-- （細かい時刻があると、名前なしの意見と照らし合わせて書いた人が分かってしまうため）
CREATE TABLE IF NOT EXISTS sends (
  member_hash TEXT NOT NULL,
  hour INTEGER NOT NULL,
  n INTEGER NOT NULL,
  PRIMARY KEY (member_hash, hour)
);

-- 「Claude と整える」の回数（1 時間ごとの合計だけ）
CREATE TABLE IF NOT EXISTS shape_uses (
  member_hash TEXT NOT NULL,
  hour INTEGER NOT NULL,
  n INTEGER NOT NULL,
  PRIMARY KEY (member_hash, hour)
);
