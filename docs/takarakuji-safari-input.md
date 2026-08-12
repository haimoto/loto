# 宝くじ公式サイトへの Safari 自動入力

`scripts/takarakuji_fill.py` の運用ドキュメント。**都度 AppleScript を書き起こさない**ためにある。

## なぜスクリプトなのか（2026-08-12 の失敗）

予測生成 → ページ probe → 関数シグネチャ調査 → 入力ロジック組み立て、と往復しているうちに
宝くじ公式サイトのセッションが切れ、**ログアウトして15組の入力が全部無駄になった**。
このページはセッションが短い。**着手から入力完了までを最短にすることが最優先**で、
丁寧な段階検証より速度が勝る。

## 使い方

```bash
python3 scripts/takarakuji_fill.py                      # loto7 を hitprob10口 + ev5口 で入力
python3 scripts/takarakuji_fill.py --hitprob 15 --ev 0  # hitprob だけ15口
python3 scripts/takarakuji_fill.py --loto loto6         # ロト6
python3 scripts/takarakuji_fill.py --dry-run            # 数字だけ（Safari を触らない）
python3 scripts/takarakuji_fill.py --check              # ログイン/締切/回号/入力済み数だけ確認
python3 scripts/takarakuji_fill.py --verify             # 入力済みの組を読み出す
python3 scripts/takarakuji_fill.py --open               # タブが無ければ開く（ログインは手動）
python3 scripts/takarakuji_fill.py --sets '1 2 3 4 5 6 7/8 9 10 11 12 13 14'
```

- `--hitprob` = 通常（命中率特化・履歴非依存）、`--ev` = 1等狙い（配当分配最適化・履歴依存）
- 既に数字が入っているページには `--force` なしでは入力しない（上書き事故の防止）
- **「カートに入れる」以降は押さない。** 購入はユーザーの操作。

## 事前確認（`--check` の読み方）

| 項目 | 意味 | 異常時 |
|---|---|---|
| `login` | `ログアウト` リンクの有無 | `no` なら中断。先に手動ログイン |
| `deadline` | 「発売締切まであと N 日」 | 締切当日は余裕を見て早く回す |
| `round` / `draw` | 回号・抽せん日 | 想定した回号か確認 |
| `grids` | `.m_lotteryNumInputNum` の数（PC/SP で2倍） | 0 ならページが読み込めていない |
| `filled` | 選択済みボタン数（`is_myself`） | 0 でなければ既に入力済み |

## ページの構造（実測）

- URL: `https://www.takarakuji-official.jp/ec/loto7/`（ロト6は `/ec/loto6/`）
- **同一ページに PC/SP の重複 DOM がある。** グリッド数・ボタン数は常に「組数×2」。
  可視判定は `offsetParent !== null`。片方を操作すれば両方に反映される。
- 申込枠は **A〜Y の25枠まで**。初期表示は A〜E の5枠で、「次の申込数字へ」を押すと枠が増える。
- 数字ボタン `.m_lotteryNumInputNum_btn` は **`.click()` では無反応**。
  `mouseover / mousedown / mouseup / click` の `MouseEvent` を順に dispatch する。選択状態は `is_myself` クラス。
- 「次の申込数字へ」= `button.m_lotteryNumInputForm_btn`。7個（ロト6は6個）選ぶまで `is_disabled` が付いていて押せない。
- 非表示グリッドも `is_myself` を保持するので、**最後に全枠を一括照合できる**。

## 落とし穴

- **背面タブだと合成クリックが巻き戻る。** SPA の描画間引きで選択が消え、
  「実行中の検証は OK なのに後から数えると 0」になる。各組の入力前に
  `activate` + `set index of window to 1` + `set current tab` を**1つの `tell` にまとめて**呼ぶ
  （3回の osascript に分けると遅く、Bash 実行でターミナルが前面を奪い返す隙ができる）。
- **`window 1` / `front window` は z-order で動く。** このMacは Safari プロファイルを複数併用しているので、
  必ず URL でタブを特定して `window id` で固定する（スクリプトの `find_tab` がやっている）。
- **クリック後の DOM 更新は非同期。** 同じ `do JavaScript` 内で結果を読むと古い要素を見て失敗に見える。
  待ち時間は advance 後 0.7 秒 / select 後 0.5 秒（`--adv-wait` / `--sel-wait` で調整可、
  不安定なら `--safe` で毎ステップ前面化）。
- **決済ページでは一切操作しない。** `/ec/cart/complete/` や `authentication.cardinalcommerce.com` に
  遷移していることがある。スクリプトは URL が `/ec/loto7/` でなければ即中断する。

## 予測の出し方だけ知りたい時

```bash
python3 -c "
from pathlib import Path
from loto_predictor_chatgpt import parse_csv, run_hitprob
d = parse_csv(Path('loto7_data.csv').read_text(), 'loto7')
run_hitprob(d, 'loto7', num_sets=10)"
```

CLI（`python3 loto_predictor_chatgpt.py loto7 loto7_data.csv hitprob`）は
引数が `loto / csv / mode` の3つだけで**組数を取れない**（5口固定）。
組数を指定するなら上のように直接呼ぶか、`scripts/takarakuji_fill.py --dry-run` を使う。

## any3 は「入賞率」ではない

`3個以上1本=78.87%` は「N口中1口でも本数字3個以上」の確率。
ロト7は本数字3個だけでは無入賞（6等は3個＋ボーナス1個、5等＝4個）。
入賞（6等以上）の確率は別計算で、20口で約62%。混同して報告しない。
