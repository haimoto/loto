"""Fast walk-forward backtest for hitprob mode vs random baseline.

Usage:
    python3 backtest_hitprob_fast.py loto6 80
    python3 backtest_hitprob_fast.py loto7 80
    python3 backtest_hitprob_fast.py loto6 80 --num-sets max

This script intentionally tests the strategy that can improve portfolio-level
hit probability: fully disjoint hitprob mode. It avoids the slower historical
heuristic modes from the legacy backtest.
"""

from __future__ import annotations

import argparse
import random
import statistics
from collections import Counter
from pathlib import Path

import loto_predictor_chatgpt as lp


def _max_disjoint_sets(loto):
    cfg = lp.LOTO_CONFIG[loto]
    lo, hi = cfg["range"]
    return (hi - lo + 1) // cfg["pick"]


def _parse_num_sets(value, loto):
    max_sets = _max_disjoint_sets(loto)
    if str(value).lower() == "max":
        return max_sets
    try:
        num_sets = int(value)
    except ValueError as exc:
        raise SystemExit("--num-sets は整数または max を指定してください") from exc
    if num_sets < 1:
        raise SystemExit("--num-sets は1以上を指定してください")
    if num_sets > max_sets:
        raise SystemExit(f"{loto} の hitprob 完全非重複上限は {max_sets} 組です")
    return num_sets


def _rand_sets(cfg, num_sets, rng):
    lo, hi = cfg["range"]
    pool = list(range(lo, hi + 1))
    return [tuple(sorted(rng.sample(pool, cfg["pick"]))) for _ in range(num_sets)]


def _hits_and_tiers(sets, draw, loto):
    w = set(draw.main)
    hits = []
    tiers = []
    for s in sets:
        hits.append(len(set(s) & w))
        tiers.append(lp.classify_prize(s, draw, loto))
    return hits, tiers


def _summary(label, totals, per_set_hits, per_set_tiers, any3_rounds, rounds, cfg, prize_avg, loto):
    hit_counter = Counter(per_set_hits)
    tier_counter = Counter(t for t in per_set_tiers if t is not None)
    print(f"[{label}]")
    print(f"  平均ヒット/回: {statistics.mean(totals):.3f}  中央値: {statistics.median(totals):.1f}")
    print(f"  平均ヒット/組: {statistics.mean(per_set_hits):.3f}")
    print("  分布: " + "  ".join(f"{k}個:{hit_counter.get(k, 0)}" for k in range(0, cfg['pick'] + 1) if hit_counter.get(k, 0)))
    tiers = cfg.get("tiers", 5)
    tier_parts = [f"{t}等:{tier_counter.get(t, 0)}" for t in range(1, tiers + 1) if tier_counter.get(t, 0)]
    win_total = sum(tier_counter.values())
    print(f"  等級内訳: {'  '.join(tier_parts) if tier_parts else '入賞なし'}  (入賞 {win_total}組, {100*win_total/len(per_set_hits):.2f}%)")
    print(f"  3個以上を1口以上含む回: {100*any3_rounds/rounds:.1f}%")
    if prize_avg:
        total_yen = sum(prize_avg.get(t, 0) * tier_counter.get(t, 0) for t in range(1, tiers + 1))
        cost = 200 if loto == "loto6" else 300
        per_set_yen = total_yen / len(per_set_hits) if per_set_hits else 0
        print(f"  賞金合計(実績平均): {total_yen:,.0f}円  1組平均: {per_set_yen:,.0f}円  ROI: {per_set_yen/cost-1:+.1%}")


def backtest(loto="loto6", rounds=80, min_history=50, seed=42, random_seeds=200, num_sets=5):
    csv_path = Path(f"{loto}_data.csv")
    draws = lp.parse_csv(csv_path.read_text(), loto)
    cfg = lp.LOTO_CONFIG[loto]
    if num_sets < 1:
        raise SystemExit("num_sets は1以上を指定してください")
    available = min(rounds, len(draws) - min_history)
    if available < 1:
        raise SystemExit(f"データ不足: {len(draws)}回")
    rounds = available

    _, gen, _ = lp.generate_hitprob_from_draws(draws[min_history:], loto, num_sets=num_sets)
    hitprob_sets = [tuple(nums) for _, nums in gen.sets]
    exact = lp.exact_hitprob(hitprob_sets, loto)
    prize_avg = lp.average_prize_yen(draws, loto)

    totals = {"hitprob": [], "random": []}
    per_set = {"hitprob": [], "random": []}
    per_tier = {"hitprob": [], "random": []}
    any3 = {"hitprob": 0.0, "random": 0.0}

    base_rng = random.Random(seed)
    for i in range(rounds):
        target = draws[i]
        h, t = _hits_and_tiers(hitprob_sets, target, loto)
        totals["hitprob"].append(sum(h))
        per_set["hitprob"].extend(h)
        per_tier["hitprob"].extend(t)
        if any(x >= 3 for x in h):
            any3["hitprob"] += 1

        seed_hits = []
        seed_tiers = []
        seed_totals = []
        seed_any3 = 0
        for _ in range(random_seeds):
            rng = random.Random(base_rng.randrange(2**31))
            sets = _rand_sets(cfg, num_sets, rng)
            rh, rt = _hits_and_tiers(sets, target, loto)
            seed_hits.extend(rh)
            seed_tiers.extend(rt)
            seed_totals.append(sum(rh))
            if any(x >= 3 for x in rh):
                seed_any3 += 1
        totals["random"].append(statistics.mean(seed_totals))
        per_set["random"].extend(seed_hits)
        per_tier["random"].extend(seed_tiers)
        any3["random"] += seed_any3 / random_seeds

    print(f"=== fast walk-forward backtest [{loto}] ===")
    print(f"対象: 直近{rounds}回  組数: {num_sets}  ランダム平均: {random_seeds}シード")
    print(f"hitprob exact: 3個以上1本={100*exact['any3']:.2f}%  4個以上1本={100*exact['any4']:.2f}%  5個以上1本={100*exact['any5']:.3f}%")
    print()
    _summary("命中率特化", totals["hitprob"], per_set["hitprob"], per_tier["hitprob"], any3["hitprob"], rounds, cfg, prize_avg, loto)
    print()
    _summary("ランダム平均", totals["random"], per_set["random"], per_tier["random"], any3["random"], rounds, cfg, prize_avg, loto)
    print()
    expected = cfg["pick"] * cfg["pick"] / (cfg["range"][1] - cfg["range"][0] + 1) * num_sets
    print(f"理論期待値: {expected:.3f}個/回（どの選び方でも同じ）")
    print("注: hitprob は合計ヒット期待値ではなく、少なくとも1口が3個以上になる確率を上げる設計です。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast hitprob backtest.")
    parser.add_argument("loto", nargs="?", choices=sorted(lp.LOTO_CONFIG), default="loto6")
    parser.add_argument("rounds", nargs="?", type=int, default=80)
    parser.add_argument("min_history", nargs="?", type=int, default=50)
    parser.add_argument(
        "--num-sets",
        default="5",
        help="購入組数。整数または max（loto6=7, loto7=5）。デフォルトは5。",
    )
    args = parser.parse_args()
    num_sets = _parse_num_sets(args.num_sets, args.loto)
    backtest(loto=args.loto, rounds=args.rounds, min_history=args.min_history, num_sets=num_sets)
