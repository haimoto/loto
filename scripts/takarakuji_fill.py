#!/usr/bin/env python3
"""宝くじ公式サイトの数字選択ページへ予測組を自動入力する常設ツール。

毎回 AppleScript を書き起こすと時間を食い、サイトのセッションが切れる。
このスクリプトは「タブ検出 → 状態確認 → 予測生成 → 入力 → 照合」を1コマンドで通す。

使い方:
    python3 scripts/takarakuji_fill.py                       # loto7 を hitprob10口 + ev5口 で入力
    python3 scripts/takarakuji_fill.py --hitprob 15 --ev 0   # hitprob だけ15口
    python3 scripts/takarakuji_fill.py --loto loto6          # ロト6
    python3 scripts/takarakuji_fill.py --dry-run             # 数字を出すだけ（Safari を触らない）
    python3 scripts/takarakuji_fill.py --check               # ページ状態だけ確認（ログイン/締切/入力済み）
    python3 scripts/takarakuji_fill.py --verify              # 入力済みの組を読み出して照合
    python3 scripts/takarakuji_fill.py --sets '1 2 3 4 5 6 7/8 9 10 11 12 13 14'

制約:
    - 「カートに入れる」以降は購入操作なので絶対に触らない（数字選択まで）。
    - 既に数字が入っているページには --force なしでは入力しない。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PAGE_PATH = {"loto6": "/ec/loto6/", "loto7": "/ec/loto7/"}
PICK = {"loto6": 6, "loto7": 7}
NUM_MAX = {"loto6": 43, "loto7": 37}
SEP = "||"

# 数字ボタンは .click() では無反応。MouseEvent 一式の dispatch が必要。
DISPATCH = (
    "['mouseover','mousedown','mouseup','click']"
    ".forEach(function(t){b.dispatchEvent(new MouseEvent(t,"
    "{bubbles:true,cancelable:true,view:window}));});"
)


# ---------------------------------------------------------------- AppleScript

def _osa(script: str) -> str:
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"osascript 失敗: {r.stderr.strip()}")
    return r.stdout.strip()


def _js_literal(js: str) -> str:
    """JS を AppleScript の文字列リテラルに埋める。JS 側はシングルクォートで書くこと。"""
    return '"' + js.replace("\\", "\\\\").replace('"', '\\"') + '"'


class Tab:
    """URL で特定した Safari タブ。window index は z-order で動くので id で固定する。"""

    def __init__(self, win_id: int, index: int, url: str, title: str):
        self.win_id = win_id
        self.index = index
        self.url = url
        self.title = title

    def __str__(self) -> str:
        return f"window id={self.win_id} tab={self.index} | {self.title}"

    def js(self, code: str) -> str:
        return _osa(
            f"tell application \"Safari\" to return (do JavaScript {_js_literal(code)} "
            f"in tab {self.index} of window id {self.win_id})"
        )

    def focus(self) -> None:
        """SPA は背面タブだと合成クリックが巻き戻る。1つの tell にまとめて呼ぶ（速度）。"""
        _osa(
            'tell application "Safari"\n'
            "  activate\n"
            f"  set index of window id {self.win_id} to 1\n"
            f"  set current tab of window id {self.win_id} to tab {self.index} of window id {self.win_id}\n"
            "end tell"
        )


def open_tab(loto: str) -> None:
    url = "https://www.takarakuji-official.jp" + PAGE_PATH[loto]
    _osa(f'tell application "Safari"\n  activate\n  tell window 1 to set current tab to (make new tab with properties {{URL:"{url}"}})\nend tell')
    time.sleep(4)


def find_tab(loto: str) -> Tab:
    needle = PAGE_PATH[loto]
    out = _osa(
        'tell application "Safari"\n'
        '  set out to ""\n'
        "  repeat with w from 1 to (count of windows)\n"
        "    repeat with t from 1 to (count of tabs of window w)\n"
        "      set u to URL of tab t of window w\n"
        f'      if u contains "{needle}" then set out to out & (id of window w) & "\t" & t '
        '& "\t" & u & "\t" & (name of tab t of window w) & linefeed\n'
        "    end repeat\n"
        "  end repeat\n"
        "  return out\n"
        "end tell"
    )
    rows = [r for r in out.splitlines() if r.strip()]
    if not rows:
        raise SystemExit(
            f"対象タブが見つからない: Safari で https://www.takarakuji-official.jp{needle} を開いてログインしておくこと"
        )
    if len(rows) > 1:
        print(f"注意: 候補タブが {len(rows)} 個。先頭を使う", file=sys.stderr)
    win, idx, url, title = rows[0].split("\t", 3)
    return Tab(int(win), int(idx), url, title)


# ------------------------------------------------------------------ ページ状態

def page_state(tab: Tab, loto: str) -> dict:
    """URL・ログイン・締切・枠数・入力済み数を1往復でまとめて取る。"""
    js = (
        "(function(){var o=[];"
        "o.push('url=' + location.href);"
        "var t=document.body.innerText;"
        "var m=t.match(/([^\\s]+)様/);o.push('user=' + (m?m[1]:'-'));"
        "var d=t.match(/発売締切まで[^\\n]*/);o.push('deadline=' + (d?d[0]:'-'));"
        "var r=t.match(/第(\\d+)回/);o.push('round=' + (r?r[1]:'-'));"
        "var s=t.match(/抽せん日：([\\d\\/]+)/);o.push('draw=' + (s?s[1]:'-'));"
        "var k=t.match(/A[~～]([A-Z])枠/);o.push('maxframe=' + (k?k[1]:'-'));"
        "o.push('grids=' + document.querySelectorAll('.m_lotteryNumInputNum').length);"
        "o.push('filled=' + document.querySelectorAll('.m_lotteryNumInputNum_btn.is_myself').length);"
        "o.push('login=' + (/ログアウト/.test(t) ? 'yes' : 'no'));"
        "return o.join('" + SEP + "');})()"
    )
    raw = tab.js(js)
    state = {}
    for kv in raw.split(SEP):
        if "=" in kv:
            k, v = kv.split("=", 1)
            state[k] = v
    return state


def assert_ready(state: dict, loto: str, force: bool) -> None:
    if PAGE_PATH[loto] not in state.get("url", ""):
        raise SystemExit(f"中断: 対象ページではない url={state.get('url')}（決済・完了ページでは一切操作しない）")
    if state.get("login") != "yes":
        raise SystemExit("中断: ログアウト状態。先にログインしてからやり直すこと")
    filled = int(state.get("filled", "0") or 0)
    if filled and not force:
        raise SystemExit(f"中断: 既に {filled // 2 // 7} 組ぶん入力済み。上書きするなら --force")


# -------------------------------------------------------------------- 組の生成

def build_sets(loto: str, hitprob: int, ev: int, csv_path: Path) -> list[list[int]]:
    from loto_predictor_chatgpt import (
        generate_from_draws,
        generate_hitprob_from_draws,
        parse_csv,
    )

    draws = parse_csv(csv_path.read_text(), loto)
    sets: list[list[int]] = []
    if hitprob:
        _, gen, _ = generate_hitprob_from_draws(draws, loto, num_sets=hitprob)
        sets += [sorted(nums) for _, nums in gen.sets]
    if ev:
        _, gen, _ = generate_from_draws(draws, loto, num_sets=ev, ev_mode=True)
        sets += [sorted(nums) for _, nums in gen.sets]
    return sets


def parse_sets(spec: str) -> list[list[int]]:
    out = []
    for chunk in spec.split("/"):
        chunk = chunk.strip().replace(",", " ")
        if chunk:
            out.append(sorted(int(x) for x in chunk.split()))
    return out


def validate(sets: list[list[int]], loto: str) -> None:
    pick, hi = PICK[loto], NUM_MAX[loto]
    for i, s in enumerate(sets, 1):
        if len(s) != pick or len(set(s)) != pick:
            raise SystemExit(f"組{i}: {loto} は重複なし{pick}個。received={s}")
        if not all(1 <= v <= hi for v in s):
            raise SystemExit(f"組{i}: 数字が 1〜{hi} の範囲外。received={s}")


# ---------------------------------------------------------------------- 入力

def _advance_js() -> str:
    return (
        "(function(){var c=Array.from(document.querySelectorAll('button.m_lotteryNumInputForm_btn'))"
        ".filter(function(e){return e.offsetParent!==null&&!/is_disabled/.test(e.className);});"
        "if(!c.length)return 'ERR:no-advance';var b=c[0];" + DISPATCH + "return 'OK';})()"
    )


def _select_js(want: list[int]) -> str:
    return (
        "(function(){var want=[" + ",".join(map(str, want)) + "];"
        "var btns=Array.from(document.querySelectorAll('.m_lotteryNumInputNum_btn'))"
        ".filter(function(e){return e.offsetParent!==null;});var n=0;"
        "want.forEach(function(v){var b=btns.filter(function(e){"
        "return parseInt(e.textContent.trim(),10)===v;})[0];"
        "if(b){" + DISPATCH + "n++;}});return String(n);})()"
    )


def _verify_js() -> str:
    """非表示グリッドも is_myself を保持するので全枠を一括で読める。"""
    return (
        "(function(){var g=Array.from(document.querySelectorAll('.m_lotteryNumInputNum'));var u=[];"
        "g.forEach(function(x){var s=Array.from(x.querySelectorAll('.m_lotteryNumInputNum_btn.is_myself'))"
        ".map(function(e){return parseInt(e.textContent.trim(),10);}).sort(function(a,b){return a-b;});"
        "if(s.length&&u.indexOf(s.join(' '))<0)u.push(s.join(' '));});"
        "return document.querySelectorAll('.m_lotteryNumInputNum_btn.is_myself').length"
        " + '" + SEP + "' + u.join('" + SEP + "');})()"
    )


def fill(tab: Tab, sets: list[list[int]], pick: int, adv_wait: float, sel_wait: float, safe: bool) -> int:
    """組ごとに advance → select。毎組の verify はしない（セッション切れ対策の速度優先）。"""
    done = 0
    for i, want in enumerate(sets):
        tab.focus()
        if i > 0:
            got = tab.js(_advance_js())
            if got != "OK":
                print(f"組{i + 1}: 枠送りに失敗 -> {got}", file=sys.stderr)
                break
            time.sleep(adv_wait)
            if safe:
                tab.focus()
        n = tab.js(_select_js(want))
        print(f"  組{i + 1:2d} {' '.join(f'{v:2d}' for v in want)} -> selected={n}")
        if n != str(pick):
            print(f"組{i + 1}: {pick}個選べていない（中断）", file=sys.stderr)
            break
        done += 1
        time.sleep(sel_wait)
    return done


def verify(tab: Tab, expected: list[list[int]] | None, pick: int) -> bool:
    raw = tab.js(_verify_js())
    parts = raw.split(SEP)
    total = int(parts[0] or 0)
    got = [list(map(int, p.split())) for p in parts[1:] if p.strip()]
    print(f"\n=== 照合 === totalMyself={total} uniqueSets={len(got)}")
    for s in got:
        print("  " + " ".join(f"{v:2d}" for v in s))
    if expected is None:
        return True
    ok = sorted(map(tuple, got)) == sorted(map(tuple, expected))
    if ok:
        print(f"OK: {len(expected)}組すべて一致（totalMyself={total} = {pick}×{len(expected)}×2レイアウト）")
    else:
        missing = [s for s in expected if s not in got]
        extra = [s for s in got if s not in expected]
        print(f"NG: 不一致 missing={missing} extra={extra}", file=sys.stderr)
    return ok


# ----------------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loto", choices=["loto6", "loto7"], default="loto7")
    ap.add_argument("--hitprob", type=int, default=10, help="通常（命中率特化）の組数")
    ap.add_argument("--ev", type=int, default=5, help="1等狙い（配当分配最適化）の組数")
    ap.add_argument("--sets", help="組を直接指定。'/' 区切り、数字は空白かカンマ区切り")
    ap.add_argument("--csv", help="既定は <loto>_data.csv")
    ap.add_argument("--dry-run", action="store_true", help="数字を出すだけ。Safari を触らない")
    ap.add_argument("--check", action="store_true", help="ページ状態だけ確認")
    ap.add_argument("--verify", action="store_true", help="入力済みの組を読み出す")
    ap.add_argument("--force", action="store_true", help="入力済みでも続行")
    ap.add_argument("--open", action="store_true", help="対象タブが無ければ Safari で開く（ログインは手動）")
    ap.add_argument("--safe", action="store_true", help="advance と select の間にも前面化を挟む（遅いが確実）")
    ap.add_argument("--adv-wait", type=float, default=0.7)
    ap.add_argument("--sel-wait", type=float, default=0.5)
    args = ap.parse_args()

    pick = PICK[args.loto]
    csv_path = Path(args.csv) if args.csv else REPO / f"{args.loto}_data.csv"

    sets: list[list[int]] = []
    if not (args.check or args.verify):
        sets = parse_sets(args.sets) if args.sets else build_sets(args.loto, args.hitprob, args.ev, csv_path)
        validate(sets, args.loto)
        print(f"[{args.loto}] {len(sets)}組"
              + (f"（hitprob {args.hitprob} + ev {args.ev}）" if not args.sets else "（--sets 指定）"))
        for i, s in enumerate(sets, 1):
            tag = "通常" if not args.sets and i <= args.hitprob else ("1等狙い" if not args.sets else "指定")
            print(f"  {i:2d} {tag:5s} " + " ".join(f"{v:2d}" for v in s))
        if args.dry_run:
            return 0

    if args.open:
        try:
            find_tab(args.loto)
        except SystemExit:
            open_tab(args.loto)
    tab = find_tab(args.loto)
    print(f"\n対象タブ: {tab}")
    state = page_state(tab, args.loto)
    print("  " + "  ".join(f"{k}={v}" for k, v in state.items()))

    if args.verify:
        return 0 if verify(tab, None, pick) else 1
    if args.check:
        return 0

    assert_ready(state, args.loto, args.force)

    t0 = time.time()
    print("\n入力開始（カートには入れない）")
    done = fill(tab, sets, pick, args.adv_wait, args.sel_wait, args.safe)
    time.sleep(1.5)
    tab.focus()
    ok = verify(tab, sets if done == len(sets) else None, pick)
    print(f"\n所要 {time.time() - t0:.1f}秒 / {done}/{len(sets)}組")
    print("※ 購入は手動。数字を確認して「カートに入れる」へ進むこと")
    return 0 if (ok and done == len(sets)) else 1


if __name__ == "__main__":
    sys.exit(main())
