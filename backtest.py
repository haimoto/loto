"""Walk-forward backtest for the loto predictor.

For each of the most recent N draws, predict using only the history that was
available before that draw, then compare against the actual winning numbers.
Compares strategies: normal, EV, greedy, relaxed, hitprob, and a random baseline.

v5.5: 等級判定を厳密化（loto6: 1〜5等、loto7: 1〜6等。ボーナス込み）。
      賞金期待値を全期間の実績平均ベースで算出。

Usage:
    python3 backtest.py                # loto6, 50 rounds
    python3 backtest.py loto6 80       # loto6, 80 rounds
    python3 backtest.py loto7 30       # loto7, 30 rounds
    python3 backtest.py loto6 80 --num-sets 7
"""

from __future__ import annotations

import argparse
import copy
import random
import statistics
import time
from collections import Counter

import loto_predictor_chatgpt as lp

def _rand_sets(cfg, num_sets, rng):
    lo, hi = cfg["range"]
    pool = list(range(lo, hi + 1))
    out = []
    for _ in range(num_sets):
        out.append(tuple(sorted(rng.sample(pool, cfg["pick"]))))
    return out


def _rand_baseline_hits(cfg, num_sets, draw, loto, rngs):
    # Average across multiple seeds so the baseline converges toward the
    # theoretical expectation rather than riding a single lucky draw.
    per_set_avg = []
    totals = []
    raw_sets = []
    raw_tiers = []
    for rng in rngs:
        sets = _rand_sets(cfg, num_sets, rng)
        h, t = _hits_and_tiers(sets, draw, loto)
        per_set_avg.append(h)
        totals.append(sum(h))
        raw_sets.extend(sets)
        raw_tiers.extend(t)
    flat_per_set = [x for sub in per_set_avg for x in sub]
    avg_total = sum(totals) / len(totals)
    return avg_total, flat_per_set, raw_sets, raw_tiers


def _predict_normal(history, loto, num_sets):
    _, gen, _ = lp.generate_from_draws(
        history, loto, num_sets=num_sets,
        params_map=lp.MODEL_PARAMS, portfolio_map=lp.PORTFOLIO_PARAMS,
        ev_mode=False,
    )
    return [tuple(nums) for _, nums in gen.sets]


def _predict_ev(history, loto, num_sets):
    _, gen, _ = lp.generate_from_draws(
        history, loto, num_sets=num_sets,
        params_map=lp.MODEL_PARAMS, portfolio_map=lp.PORTFOLIO_PARAMS,
        ev_mode=True,
    )
    return [tuple(nums) for _, nums in gen.sets]


def _predict_greedy(history, loto, num_sets):
    # Pure score-driven selection (no portfolio coverage diversification).
    _, gen, _ = lp.generate_from_draws(
        history, loto, num_sets=num_sets,
        params_map=lp.MODEL_PARAMS, portfolio_map=lp.PORTFOLIO_PARAMS,
        selection_mode="greedy", ev_mode=False,
    )
    return [tuple(nums) for _, nums in gen.sets]


def _predict_hitprob(history, loto, num_sets):
    # Coverage-first, overlap-minimized portfolio. Objective is increasing the
    # probability that at least one set hits >=3, not raising total hits.
    _, gen, _ = lp.generate_hitprob_from_draws(
        history, loto, num_sets=num_sets,
        params_map=lp.MODEL_PARAMS, portfolio_map=lp.PORTFOLIO_PARAMS,
    )
    return [tuple(nums) for _, nums in gen.sets]


def _predict_relaxed(history, loto, num_sets):
    # Same as normal but with looser inter-set overlap and weaker coverage penalty.
    mp = copy.deepcopy(lp.MODEL_PARAMS)
    pp = copy.deepcopy(lp.PORTFOLIO_PARAMS)
    mp[loto]["max_overlap"] = 3
    pp[loto]["new_num"] *= 0.5
    pp[loto]["new_pair"] *= 0.5
    pp[loto]["num_repeat"] *= 0.5
    pp[loto]["num_repeat_hard"] *= 0.5
    pp[loto]["min_unique_target"] = 14
    pp[loto]["min_unique_bonus"] = 0.05
    _, gen, _ = lp.generate_from_draws(
        history, loto, num_sets=num_sets,
        params_map=mp, portfolio_map=pp, ev_mode=False,
    )
    return [tuple(nums) for _, nums in gen.sets]


def _hits(sets, winning):
    w = set(winning)
    return [len(set(s) & w) for s in sets]


def _hits_and_tiers(sets, draw, loto):
    w = set(draw.main)
    hits = []
    tiers = []
    for s in sets:
        hits.append(len(set(s) & w))
        tiers.append(lp.classify_prize(s, draw, loto))
    return hits, tiers


def _summary(label, per_round_total, per_set_hits, per_set_tiers, cfg,
             prize_avg, loto, ev_proxy=None, num_sets=None, random_seeds=1):
    n = len(per_round_total)
    avg_total = statistics.mean(per_round_total)
    med_total = statistics.median(per_round_total)
    avg_per_set = statistics.mean(per_set_hits)
    hit_counter = Counter(per_set_hits)
    if num_sets is None:
        num_sets = int(len(per_set_hits) / n) if n else 0
    seed_label = f" × {random_seeds}シード" if random_seeds > 1 else ""
    print(f"[{label}]")
    print(f"  平均ヒット/回: {avg_total:.2f}  中央値: {med_total:.1f}  評価: {n}回 × {num_sets}組{seed_label} = {len(per_set_hits)}組")
    print(f"  平均ヒット/組: {avg_per_set:.3f}")
    hist_parts = [f"{k}個:{hit_counter.get(k, 0)}" for k in range(0, 8) if hit_counter.get(k, 0)]
    print(f"  分布: {'  '.join(hist_parts)}")
    # 等級別カウント（厳密判定: loto6=2等はボーナス1致、loto7=2/6等はボーナス1or2）
    tier_counter = Counter(t for t in per_set_tiers if t is not None)
    tiers = cfg.get("tiers", 5)
    tier_parts = []
    for t in range(1, tiers + 1):
        c = tier_counter.get(t, 0)
        if c:
            tier_parts.append(f"{t}等:{c}")
    win_total = sum(tier_counter.values())
    print(f"  等級内訳: {'  '.join(tier_parts) if tier_parts else '入賞なし'}  "
          f"(入賞 {win_total}組, {100*win_total/len(per_set_hits):.2f}%)")
    # 賞金期待値（実績平均 × 各等級の的中口数）
    if prize_avg:
        total_yen = sum(prize_avg.get(t, 0) * tier_counter.get(t, 0) for t in range(1, tiers + 1))
        per_round_yen = total_yen / n if n else 0
        per_set_yen = total_yen / len(per_set_hits) if per_set_hits else 0
        cost = 200 if loto == "loto6" else 300
        roi = per_set_yen / cost - 1 if cost else 0
        print(f"  賞金合計(実績平均): {total_yen:>15,.0f}円  "
              f"1回平均: {per_round_yen:>11,.0f}円  "
              f"1組平均: {per_set_yen:>9,.0f}円  ROI: {roi:+.1%}")
    if ev_proxy:
        ht = cfg.get("high_threshold", 31)
        print(f"  EV不人気スコア平均: {ev_proxy['ev_avg']:.2f}  "
              f"高位(>{ht})平均: {ev_proxy['high_avg']:.2f}個/組  "
              f"日付内のみ率: {100*ev_proxy['date_only_rate']:.1f}%  "
              f"合計平均: {ev_proxy['sum_avg']:.1f}")


def _ev_proxy(all_sets, cfg):
    # Proxy for expected payout share under the EV-mode objective: EV score,
    # number of >=high_threshold picks, and avoidance of "all <= 31" combos
    # all correlate with less crowded jackpot tiers.
    ht = cfg.get("high_threshold", 31)
    ev_scores = [lp._ev_unpopularity(s, cfg) for s in all_sets]
    highs = [sum(1 for n in s if n > ht) for s in all_sets]
    date_only = sum(1 for s in all_sets if all(n <= 31 for n in s))
    sums = [sum(s) for s in all_sets]
    return {
        "ev_avg": statistics.mean(ev_scores),
        "high_avg": statistics.mean(highs),
        "date_only_rate": date_only / len(all_sets),
        "sum_avg": statistics.mean(sums),
    }


def _paired_sign_test(a, b):
    # Paired comparison: for each round i, which strategy scored more hits?
    # Returns (wins_a, losses_a, ties, two_sided_p_rough).
    assert len(a) == len(b)
    wins = sum(1 for x, y in zip(a, b) if x > y)
    losses = sum(1 for x, y in zip(a, b) if x < y)
    ties = len(a) - wins - losses
    # Sign-test p-value via binomial tail with n = wins + losses.
    n = wins + losses
    if n == 0:
        return wins, losses, ties, 1.0
    k = min(wins, losses)
    # Two-sided p-value approximated from binomial CDF tail.
    from math import comb
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    return wins, losses, ties, p


def backtest(loto="loto6", rounds=50, min_history=50, seed=42, random_seeds=20, num_sets=5):
    csv_path = f"{loto}_data.csv"
    with open(csv_path) as f:
        all_draws = lp.parse_csv(f.read(), loto)
    cfg = lp.LOTO_CONFIG[loto]
    if num_sets < 1:
        raise SystemExit("num_sets は1以上を指定してください")

    if len(all_draws) < min_history + 1:
        print(f"データ不足: {len(all_draws)}回（最低{min_history + 1}回必要）")
        return

    available = min(rounds, len(all_draws) - min_history)
    if available < rounds:
        print(f"指定{rounds}回 → 利用可能{available}回に縮小")
        rounds = available

    base_rng = random.Random(seed)
    prize_avg = lp.average_prize_yen(all_draws, loto)
    print(f"\n=== walk-forward backtest [{loto}] ===")
    print(f"対象: 直近{rounds}回  組数: {num_sets}  学習履歴: 各ターゲット以前の全データ（最大{lp.LOOKBACK}回）")
    print(f"ランダムベースライン: {random_seeds}シード平均")
    if prize_avg:
        ptxt = "  ".join(f"{t}等={y:,.0f}円" for t, y in prize_avg.items())
        print(f"等級別実績平均賞金: {ptxt}")
    print()

    modes = ["normal", "ev", "greedy", "relaxed", "hitprob", "random"]
    totals = {m: [] for m in modes}
    per_set = {m: [] for m in modes}
    per_tier = {m: [] for m in modes}
    all_sets = {m: [] for m in modes}
    any3_rounds = {m: 0 for m in modes}
    start = time.time()

    for i in range(rounds):
        target = all_draws[i]
        history = all_draws[i + 1:]
        if len(history) < min_history:
            break

        rngs = [random.Random(base_rng.randrange(2**31)) for _ in range(random_seeds)]
        rand_total, rand_flat, rand_raw, rand_tiers = _rand_baseline_hits(
            cfg, num_sets, target, loto, rngs
        )
        results = {
            "normal": _predict_normal(history, loto, num_sets),
            "ev": _predict_ev(history, loto, num_sets),
            "greedy": _predict_greedy(history, loto, num_sets),
            "relaxed": _predict_relaxed(history, loto, num_sets),
            "hitprob": _predict_hitprob(history, loto, num_sets),
        }
        for m in ("normal", "ev", "greedy", "relaxed", "hitprob"):
            hits, tiers = _hits_and_tiers(results[m], target, loto)
            totals[m].append(sum(hits))
            per_set[m].extend(hits)
            per_tier[m].extend(tiers)
            all_sets[m].extend(results[m])
            if any(h >= 3 for h in hits):
                any3_rounds[m] += 1
        totals["random"].append(rand_total)
        per_set["random"].extend(rand_flat)
        per_tier["random"].extend(rand_tiers)
        all_sets["random"].extend(rand_raw)
        # For the random baseline, count any3 across the averaged seeds.
        # rand_flat has random_seeds * num_sets entries; chunk per seed to match.
        for k in range(random_seeds):
            chunk = rand_flat[k * num_sets:(k + 1) * num_sets]
            if any(h >= 3 for h in chunk):
                any3_rounds["random"] += 1 / random_seeds

        if (i + 1) % 10 == 0 or i == rounds - 1:
            elapsed = time.time() - start
            means = {m: statistics.mean(totals[m]) for m in modes}
            print(f"  進捗 {i+1}/{rounds}  {elapsed:.1f}s  "
                  + "  ".join(f"{m}={means[m]:.2f}" for m in modes))

    print()
    labels = {"normal": "通常モード", "ev": "EV特化", "greedy": "greedy(純スコア)",
              "relaxed": "制約緩和", "hitprob": "命中率特化", "random": "ランダム"}
    for m in modes:
        proxy = _ev_proxy(all_sets[m], cfg)
        _summary(labels[m], totals[m], per_set[m], per_tier[m], cfg,
                 prize_avg, loto, ev_proxy=proxy, num_sets=num_sets,
                 random_seeds=random_seeds if m == "random" else 1)
        print()

    expected = cfg["pick"] * cfg["pick"] / (cfg["range"][1] - cfg["range"][0] + 1) * num_sets
    print(f"理論期待値: {num_sets}組合計 {expected:.2f}個/回  (1組あたり {expected/num_sets:.3f}個)")
    print()

    print("【≥3個を1口以上含む回の割合（hitprob モードの設計目標）】")
    for m in modes:
        rate = any3_rounds[m] / rounds
        print(f"  {labels[m]:>14s}: {100*rate:.1f}%")
    print()

    # Paired sign test against the averaged random baseline. Note the random
    # series is a 20-seed mean per round so its variance is ~1/20 of a single
    # draw; a Welch t on that series would exaggerate significance. A paired
    # sign test is unit-agnostic and correctly paired round-by-round.
    print("【ランダム平均との符号検定（rounds数が小さいと検出力は低い）】")
    print(f"  n={rounds}, H0: 各回の合計ヒット分布がランダム平均と同じ")
    for m in ("normal", "ev", "greedy", "relaxed", "hitprob"):
        w, losses, ties, p = _paired_sign_test(totals[m], totals["random"])
        print(f"  {labels[m]:>14s}: 勝{w} 負{losses} 分{ties}  two-sided p≈{p:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward backtest for loto strategies.")
    parser.add_argument("loto", nargs="?", choices=sorted(lp.LOTO_CONFIG), default="loto6")
    parser.add_argument("rounds", nargs="?", type=int, default=50)
    parser.add_argument("min_history", nargs="?", type=int, default=50)
    parser.add_argument("--num-sets", type=int, default=5, help="購入組数。デフォルトは5。")
    args = parser.parse_args()
    backtest(loto=args.loto, rounds=args.rounds, min_history=args.min_history, num_sets=args.num_sets)
