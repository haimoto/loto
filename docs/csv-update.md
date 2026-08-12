# CSV 更新手順（loto6_data.csv / loto7_data.csv）

**探し直さない。** データソースは確定済み。以下をそのまま使う。

## 1. 本数字＋ボーナス — 楽天×宝くじ lastresults（WebFetch、直近10回）

- ロト6: `https://takarakuji.rakuten.co.jp/backnumber/loto6/lastresults/`
- ロト7: `https://takarakuji.rakuten.co.jp/backnumber/loto7/lastresults/`

軽量な popup 版で賞金・口数・キャリーオーバーは含まれない。既存 CSV の最新回と重なる回で照合できる。

## 2. 賞金・口数・キャリーオーバー — 福井新聞ONLINE（記事本文を WebFetch）

WebSearch（`allowed_domains: ["fukuishimbun.co.jp", "oricon.co.jp"]`）で
`ロト6 第NNNN回 当選番号 1等 2等 3等 4等 5等 口数 キャリーオーバー` を検索して記事URLを得る。

⚠️ **WebSearch のサマリはキャリーオーバーを落とす**（2026-08-12 は4回とも欠落した）。
記事URL（`fukuishimbun.co.jp/articles/-/NNNNNNN`）を **WebFetch で直接開いて本文から拾う**。
本数字もここに載っているので、楽天との二重照合を兼ねる。

## 3. 避けるソース

| ソース | 理由 |
|---|---|
| みずほ銀行 | Akamai でブロック（403 / Access Denied） |
| 楽天の個別回ページ `/backnumber/loto6/{回号}/` | 422 |
| takarakuji-loto.jp `tousenp.html` | 当選数字が画像 |
| takarakuji-loto.jp `loto6_table-print.html` | ボーナスを太字で表現しておりテキストから区別できない |
| sumaispring.com / loto-7.net / loto7.thoth.jp / lottery.dmkt-sp.jp | DNS 解決失敗 |
| takarakujinet.co.jp | SSL 証明書エラー |

## 4. CSV フォーマット

```
ロト6: 回号,抽選日,n1..n6,bonus,p1_cnt..p5_cnt,p1_yen..p5_yen,carry          (20列)
ロト7: 回号,抽選日,n1..n7,bonus1,bonus2,p1_cnt..p6_cnt,p1_yen..p6_yen,carry  (24列)
```

- **降順**（最新が先頭）。ヘッダ直後に挿入する
- 日付は `YYYY/MM/DD`（ゼロパディングあり）
- 1等該当なしは `p1_cnt=0, p1_yen=0`
- 抽せん日: ロト6=月・木、ロト7=金

キャリーオーバーの整合性は自分で確認する。1等が上限額（ロト6=6億、ロト7=12億）で
打ち止めになった回は、超過分が次回へ繰り越されて `carry` が残る。
「1等が出たのに carry > 0」は矛盾ではない。

## 5. 追記後の機械検証（必須）

```bash
python3 scripts/verify_csv.py
```

列数・回号の降順連番・重複・本数字の昇順/範囲/重複・全数値列のパース・`parse_csv` 読み込みを検証する。
OK が出てから commit する。
