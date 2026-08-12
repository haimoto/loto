#!/usr/bin/env python3
"""loto6_data.csv / loto7_data.csv の機械検証。追記したら commit 前に必ず走らせる。

    python3 scripts/verify_csv.py            # 両方
    python3 scripts/verify_csv.py loto7      # 片方だけ

検証項目: 列数 / 回号の降順・連番・重複なし / 本数字の昇順・範囲・重複なし /
本数字とボーナスの重複なし / 全数値列が int / parse_csv で読めること。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SPECS = {
    "loto6": dict(ncol=20, pick=6, bonus=1, hi=43),
    "loto7": dict(ncol=24, pick=7, bonus=2, hi=37),
}


def check(loto: str) -> list[str]:
    spec = SPECS[loto]
    path = REPO / f"{loto}_data.csv"
    rows = list(csv.reader(path.read_text().splitlines()))
    header, body = rows[0], rows[1:]
    errs: list[str] = []

    if len(header) != spec["ncol"]:
        errs.append(f"ヘッダ列数 {len(header)} != {spec['ncol']}")

    nums = [int(r[0]) for r in body]
    if nums != sorted(nums, reverse=True):
        errs.append("回号が降順でない")
    if len(set(nums)) != len(nums):
        errs.append("回号が重複している")
    gaps = [(a, b) for a, b in zip(nums, nums[1:]) if a - b != 1]
    if gaps:
        errs.append(f"回号が連番でない: {gaps[:5]}")

    for r in body:
        if len(r) != spec["ncol"]:
            errs.append(f"第{r[0]}回 列数 {len(r)} != {spec['ncol']}")
            continue
        try:
            [int(v) for v in r[2:]]
        except ValueError as e:
            errs.append(f"第{r[0]}回 数値でない列がある: {e}")
            continue
        main = [int(x) for x in r[2:2 + spec["pick"]]]
        bonus = [int(x) for x in r[2 + spec["pick"]:2 + spec["pick"] + spec["bonus"]]]
        if main != sorted(main):
            errs.append(f"第{r[0]}回 本数字が昇順でない")
        if len(set(main)) != spec["pick"]:
            errs.append(f"第{r[0]}回 本数字が重複")
        if not all(1 <= v <= spec["hi"] for v in main + bonus):
            errs.append(f"第{r[0]}回 数字が 1〜{spec['hi']} の範囲外")
        if len(set(main + bonus)) != len(main + bonus):
            errs.append(f"第{r[0]}回 本数字とボーナスが重複")

    from loto_predictor_chatgpt import parse_csv

    draws = parse_csv(path.read_text(), loto)
    if not draws:
        errs.append("parse_csv が0件を返した")
    elif draws[0].number != nums[0]:
        errs.append(f"parse_csv の先頭 第{draws[0].number}回 が CSV 先頭 第{nums[0]}回 と違う")

    head = f"{path.name}: {len(body)}行 最新=第{nums[0]}回({body[0][1]}) 最古=第{nums[-1]}回 parse_csv={len(draws)}件"
    print(f"{head} -> {'OK' if not errs else 'NG'}")
    for e in errs[:20]:
        print(f"    {e}")
    return errs


def main() -> int:
    targets = sys.argv[1:] or list(SPECS)
    bad = 0
    for loto in targets:
        if loto not in SPECS:
            print(f"不明な loto: {loto}", file=sys.stderr)
            return 2
        bad += len(check(loto))
    print("ALL OK" if not bad else f"検証エラー {bad} 件")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
