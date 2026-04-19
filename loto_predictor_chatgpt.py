"""
ロト6/ロト7 予測スクリプト v5.4

設計の前提（重要）:
- ロト6/7 は独立抽選のため、過去データから「的中率」を改善することは
  数学的に不可能。HOT/COLD/ペア頻度/overdue などの統計特徴は独立試行に
  対して予測信号にならない（実測でもランダム以下）。
- 改善可能なのは (a) 当選時の配当分配（ev モード）、および
  (b) 5口のうち少なくとも1口が3個以上に届く確率（hitprob モード）。
  どちらも5口合計の期待ヒット数は変えられない（独立試行のため）。

v5.4（2026-04-19 Codex レビュー対応）の変更:
- hitprob を真の最適化へ: ランダムサンプル + greedy の近似を、履歴非依存の
  backtracking 完全非重複ポートフォリオ構築に置換。loto6/7 とも常に完全
  disjoint（loto7: union=35, loto6: union=30）を返す。
- 確率表示を exact へ: Monte Carlo（trials=10万）を全組合せ列挙による
  exact_hitprob に置換（loto6: C(43,6)=6M, loto7: C(37,7)=10M）。
  estimate_hitprob は MC 版として残存（back-compat）。
- 実測値（exact）:
  - loto7: coverage any3=40.9017% → hitprob 50.9627%（+10.06pt）
  - loto6: coverage any3=12.4266% → hitprob 13.5171%（+1.09pt）
- hitprob は履歴完全非依存（shape_score から sum_bin 除去、seed=0 固定）。
- backtest.py: Welch t を paired sign test に置換（random baseline が
  20シード平均で分散 1/20 のため t 値が誤差過大だった）。
- backtest.py: 「3個以上（入賞相当）」→「本数字3個以上」に訂正（loto7 の
  6等は本数字3個+ボーナスのため「入賞相当」は誤り）。
- EV 表示の high_threshold を動的化（ロト6=>31, ロト7=>25）。
- CSV パーサーを n1..n{pick} 明示読取 + 範囲チェックに変更。ボーナス列
  つき公式CSV も扱える。
- dead code 削除（MODEL_PARAMS_REOPT / PORTFOLIO_PARAMS_REOPT /
  PRIZE_THRESHOLDS）。

v5.3（2026-04-19）: 命中率特化モード追加（greedy近似版、v5.4 で置換済）。
v5.2（2026-04-19）: EV 特化の高位偏重を是正（high_count キャップ + 過剰集中
  ペナルティ）。
v5.1（2026-04-16）: 誠実性リファクタ。ev_weight 削除、fully_satisfied 修正、
  高位閾値の range 依存化。

インターフェース:
    parse_csv(text, loto)
    run(draws, loto, num_sets=5, ev_mode=True)          # coverage / ev
    run_hitprob(draws, loto, num_sets=5, method="exact") # hitprob
    generate_from_draws(draws, loto, num_sets=5, ev_mode=True)
    generate_hitprob_from_draws(draws, loto, num_sets=5)
    exact_hitprob(portfolio, loto)                      # 全組合せ列挙
    estimate_hitprob(portfolio, loto, trials=100000)    # Monte Carlo (back-compat)
    compare_coverage_vs_hitprob(draws, loto, num_sets=5, method="exact")

CLI:
    python3 loto_predictor_chatgpt.py <loto6|loto7> <csv> <hitprob|coverage|ev|compare>

外部ライブラリ不要。hitprob は exact 確率ベース、全モード決定論的生成。
"""

from __future__ import annotations

import random
import sys as _sys
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

# -- 定数 --

LOTO_CONFIG = {
    "loto6": {
        "range": (1, 43),
        "pick": 6,
        "odd_even_base": (3, 3),
        "small_max": 21,
        "sum_mid": (120, 160),
        "url_path": "loto6",
        # High-number bonus threshold. loto6 has 12 numbers > 31 (32-43),
        # which matches the "above calendar-date" concept.
        "high_threshold": 31,
    },
    "loto7": {
        "range": (1, 37),
        "pick": 7,
        "odd_even_base": (4, 3),
        "small_max": 18,
        "sum_mid": (125, 155),
        "url_path": "loto7",
        # loto7 only has 6 numbers > 31, so ">31" over-constrains EV sampling.
        # Use >25 instead — gives 12 high numbers, proportional to loto6.
        "high_threshold": 25,
    },
}

# 200回データまで扱えるように拡張
LOOKBACK = 200
RECENT = 12

# v4.1 の再構成ベースライン
MODEL_PARAMS_BASELINE = {
    "loto6": {
        "lookback": 24,
        "recent": 10,
        "decay_base": 0.90,
        "pair_decay": 0.94,
        "triple_decay": 0.96,
        "top_m": 16,
        "max_pool": 20,
        "extra_overdue": 4,
        "extra_cold": 0,
        "max_overlap": 2,
        "hot_pct": 0.18,
        "cold_pct": 0.18,
        "hot_limit": 2,
        "use_triple": False,
        "pool_keep_per_pool": 500,
        "num_weights": {
            "full": 0.6,
            "recent": 0.4,
            "decay": 0.8,
            "gap": 0.0,
            "due": 0.6,
            "overdue": 0.2,
            "carry": 0.2,
            "adj": 0.2,
            "bridge": 0.4,
            "cycle": 0.0,
        },
        "set_weights": {
            "pair_mean": 0.7,
            "pair_max": 0.4,
            "triple_mean": 0.0,
            "triple_max": 0.0,
            "consec": 0.5,
            "odd": 0.9,
            "low": 0.7,
            "sum": 0.9,
            "repeat": 0.0,
            "adj": 0.2,
            "span": 0.0,
            "cycle": 0.0,
            "tail": 0.0,
        },
    },
    "loto7": {
        "lookback": 22,
        "recent": 8,
        "decay_base": 0.90,
        "pair_decay": 0.94,
        "triple_decay": 0.96,
        "top_m": 16,
        "max_pool": 20,
        "extra_overdue": 4,
        "extra_cold": 0,
        "max_overlap": 2,
        "hot_pct": 0.12,
        "cold_pct": 0.12,
        "hot_limit": 2,
        "use_triple": False,
        "pool_keep_per_pool": 500,
        "num_weights": {
            "full": 0.6,
            "recent": 0.4,
            "decay": 0.8,
            "gap": 0.0,
            "due": 0.6,
            "overdue": 0.2,
            "carry": 0.2,
            "adj": 0.2,
            "bridge": 0.4,
            "cycle": 0.0,
        },
        "set_weights": {
            "pair_mean": 0.8,
            "pair_max": 0.4,
            "triple_mean": 0.0,
            "triple_max": 0.0,
            "consec": 0.5,
            "odd": 0.8,
            "low": 0.6,
            "sum": 0.7,
            "repeat": 0.3,
            "adj": 0.3,
            "span": 0.0,
            "cycle": 0.0,
            "tail": 0.0,
        },
    },
}

# 200回データで再調整した最終版デフォルト
MODEL_PARAMS = {
    "loto6": {
        "lookback": 96,
        "recent": 12,
        "decay_base": 0.94,
        "pair_decay": 0.96,
        "triple_decay": 0.97,
        "top_m": 18,
        "max_pool": 20,
        "extra_overdue": 5,
        "extra_cold": 2,
        "max_overlap": 2,
        "hot_pct": 0.14,
        "cold_pct": 0.16,
        "hot_limit": 2,
        "use_triple": False,
        "pool_keep_per_pool": 550,
        "num_weights": {
            "full": 0.72,
            "recent": 0.46,
            "decay": 0.98,
            "gap": 0.0,
            "due": 0.62,
            "overdue": 0.30,
            "carry": 0.08,
            "adj": 0.16,
            "bridge": 0.34,
            "cycle": 0.08,
        },
        "set_weights": {
            "pair_mean": 0.62,
            "pair_max": 0.24,
            "triple_mean": 0.0,
            "triple_max": 0.0,
            "consec": 0.42,
            "odd": 0.72,
            "low": 0.58,
            "sum": 0.62,
            "repeat": 0.10,
            "adj": 0.18,
            "span": 0.16,
            "cycle": 0.08,
            "tail": 0.10,
        },
    },
    "loto7": {
        "lookback": 108,
        "recent": 12,
        "decay_base": 0.94,
        "pair_decay": 0.96,
        "triple_decay": 0.97,
        "top_m": 18,
        "max_pool": 20,
        "extra_overdue": 5,
        "extra_cold": 2,
        "max_overlap": 2,
        "hot_pct": 0.12,
        "cold_pct": 0.14,
        "hot_limit": 2,
        "use_triple": False,
        "pool_keep_per_pool": 600,
        "num_weights": {
            "full": 0.70,
            "recent": 0.50,
            "decay": 1.02,
            "gap": 0.0,
            "due": 0.68,
            "overdue": 0.26,
            "carry": 0.10,
            "adj": 0.18,
            "bridge": 0.42,
            "cycle": 0.10,
        },
        "set_weights": {
            "pair_mean": 0.70,
            "pair_max": 0.28,
            "triple_mean": 0.0,
            "triple_max": 0.0,
            "consec": 0.44,
            "odd": 0.64,
            "low": 0.52,
            "sum": 0.58,
            "repeat": 0.14,
            "adj": 0.22,
            "span": 0.18,
            "cycle": 0.10,
            "tail": 0.12,
        },
    },
}

# ポートフォリオ選択の重み（coverage mode）
PORTFOLIO_PARAMS_BASELINE = {
    "loto6": {
        "indiv": 0.72,
        "new_num": 1.05,
        "new_pair": 0.30,
        "num_repeat": 0.20,
        "num_repeat_hard": 0.55,
        "new_lowbin": 0.18,
        "new_sumbin": 0.22,
        "new_spanbin": 0.10,
        "min_unique_target": 18,
        "min_unique_bonus": 0.18,
        "beam_width": 10,
        "expand_per_state": 100,
        "candidate_keep": 1200,
        "pool_size": 18,
    },
    "loto7": {
        "indiv": 0.68,
        "new_num": 1.15,
        "new_pair": 0.34,
        "num_repeat": 0.22,
        "num_repeat_hard": 0.62,
        "new_lowbin": 0.18,
        "new_sumbin": 0.24,
        "new_spanbin": 0.12,
        "min_unique_target": 22,
        "min_unique_bonus": 0.22,
        "beam_width": 10,
        "expand_per_state": 120,
        "candidate_keep": 1400,
        "pool_size": 18,
    },
}

PORTFOLIO_PARAMS = {
    "loto6": {
        "indiv": 0.68,
        "new_num": 1.18,
        "new_pair": 0.34,
        "num_repeat": 0.22,
        "num_repeat_hard": 0.72,
        "new_lowbin": 0.22,
        "new_sumbin": 0.26,
        "new_spanbin": 0.14,
        "min_unique_target": 19,
        "min_unique_bonus": 0.24,
        "beam_width": 12,
        "expand_per_state": 110,
        "candidate_keep": 1400,
        "pool_size": 18,
    },
    "loto7": {
        "indiv": 0.62,
        "new_num": 1.32,
        "new_pair": 0.38,
        "num_repeat": 0.24,
        "num_repeat_hard": 0.84,
        "new_lowbin": 0.24,
        "new_sumbin": 0.30,
        "new_spanbin": 0.16,
        "min_unique_target": 24,
        "min_unique_bonus": 0.30,
        "beam_width": 14,
        "expand_per_state": 130,
        "candidate_keep": 1600,
        "pool_size": 18,
    },
}

# 過学習チェック後の採用デフォルト: ベースライン参数 + 修正版 coverage を採用
MODEL_PARAMS = deepcopy(MODEL_PARAMS_BASELINE)
PORTFOLIO_PARAMS = deepcopy(PORTFOLIO_PARAMS_BASELINE)

_LAST_MODEL_CACHE = {}

# -- データクラス --

@dataclass
class Draw:
    number: int
    date: str
    main: tuple[int, ...]


@dataclass
class NumberStats:
    score: int
    category: Literal["HOT", "MID", "COLD"]


@dataclass
class GenerateResult:
    sets: list[tuple[str, tuple[int, ...]]]
    fully_satisfied: bool


# -- CSV パーサー --

def parse_csv(text: str, loto: str) -> list[Draw]:
    """Parse draw CSV. Reads only the first `pick` numeric columns after
    (draw_number, date) so trailing bonus columns are ignored cleanly.

    Validates: exact pick count, no duplicates, all numbers within range.
    Invalid rows are silently skipped.
    """
    cfg = LOTO_CONFIG[loto]
    pick = cfg["pick"]
    lo, hi = cfg["range"]
    draws = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or any(h in line for h in ["回号", "抽選日", "draw", "date", "#"]):
            continue
        parts = [p.strip() for p in (line.split("\t") if "\t" in line else line.split(","))]
        if len(parts) < 2 + pick:
            continue
        num_match = "".join(c for c in parts[0] if c.isdigit())
        if not num_match:
            continue
        draw_num = int(num_match)
        draw_date = parts[1].strip()
        numbers = []
        valid = True
        for token in parts[2:2 + pick]:
            tok = token.strip().strip("()")
            if not tok.isdigit():
                valid = False
                break
            n = int(tok)
            if not (lo <= n <= hi):
                valid = False
                break
            numbers.append(n)
        if not valid:
            continue
        if len(set(numbers)) != pick:
            continue
        draws.append(Draw(draw_num, draw_date, tuple(sorted(numbers))))
    draws.sort(key=lambda d: d.number, reverse=True)
    seen = set()
    unique = []
    for d in draws:
        if d.number not in seen:
            seen.add(d.number)
            unique.append(d)
    return unique[:LOOKBACK]


# -- 汎用ヘルパー --

def _params_map(override=None):
    return override if override is not None else MODEL_PARAMS


def _portfolio_map(override=None):
    return override if override is not None else PORTFOLIO_PARAMS


def _cache_key(loto, params_map, selection_mode, portfolio_map):
    return (
        loto,
        selection_mode,
        tuple(sorted((k, repr(v)) for k, v in params_map[loto].items())),
        tuple(sorted((k, repr(v)) for k, v in portfolio_map[loto].items())),
    )


def _loto_from_range(num_range):
    for loto, cfg in LOTO_CONFIG.items():
        if cfg["range"] == num_range:
            return loto
    raise ValueError(f"unknown range: {num_range}")


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _pstdev(values):
    if not values:
        return 0.0
    mu = _mean(values)
    return (sum((v - mu) ** 2 for v in values) / len(values)) ** 0.5


def _median(values):
    if not values:
        return 0.0
    arr = sorted(values)
    m = len(arr) // 2
    if len(arr) % 2:
        return arr[m]
    return (arr[m - 1] + arr[m]) / 2


def _adjacent_from_numbers(numbers, num_range):
    lo, hi = num_range
    adj = set()
    base = set(numbers)
    for n in base:
        if n - 1 >= lo:
            adj.add(n - 1)
        if n + 1 <= hi:
            adj.add(n + 1)
    return adj - base


def _pair_key(a, b):
    return (a, b) if a < b else (b, a)


def _safe_mode(values, default=0):
    if not values:
        return default
    return Counter(values).most_common(1)[0][0]


def _scaled_scores(composite):
    min_s = min(composite.values())
    max_s = max(composite.values())
    if max_s - min_s < 1e-9:
        return {n: 50 for n in composite}
    return {
        n: int(round(10 + 90 * (composite[n] - min_s) / (max_s - min_s)))
        for n in composite
    }


def _zbin(value, mean, sd):
    sd = sd or 1.0
    z = (value - mean) / sd
    if z <= -1.25:
        return -2
    if z <= -0.40:
        return -1
    if z < 0.40:
        return 0
    if z < 1.25:
        return 1
    return 2


def _ev_unpopularity(nums, cfg):
    # Humans systematically over-pick calendar dates (1-31), months (1-12), lucky digits,
    # and patterned sequences. Combinations that avoid those shapes share the jackpot
    # with fewer winners, so a higher score here means a larger payout when it hits.
    lo, hi = cfg["range"]
    n_pick = len(nums)
    # high_threshold is range-dependent: loto6=>31 (12 high nums), loto7=>25 (12 high nums).
    # Using a fixed >31 on loto7 leaves only 6 eligible numbers and over-constrains EV.
    high_threshold = cfg.get("high_threshold", 31)
    score = 0.0

    # v5.2: cap linear reward and penalize over-concentration.
    # Rewarding high_count linearly up to pick drove ロト7 EV-optimal sets to
    # cluster 6/7 numbers in 26-37, which is itself a recognizable pattern.
    # Peak reward is at pick//2+1 high numbers; beyond pick-2 a penalty kicks in.
    high_count = sum(1 for n in nums if n > high_threshold)
    high_cap = n_pick // 2 + 1
    over_cluster_limit = n_pick - 2
    score += min(high_count, high_cap) * 1.2
    if high_count > over_cluster_limit:
        score -= (high_count - over_cluster_limit) * 0.8

    # "Date-only" layer (1-31) is always an absolute human bias — birthday/anniversary picks.
    # Regardless of loto range, combinations entirely within 1-31 share the prize pool
    # with the largest group of players.
    if all(n <= 31 for n in nums):
        score -= 2.5

    month = sum(1 for n in nums if 1 <= n <= 12)
    if month >= 3:
        score -= (month - 2) * 0.8

    unlucky = sum(1 for n in nums if n in (4, 9))
    score += unlucky * 0.6

    s = sum(nums)
    # Sum thresholds scale with pick size (loto6 pick=6, loto7 pick=7).
    sum_low, sum_high = (100, 130) if n_pick == 6 else (120, 150)
    sum_tail = 150 if n_pick == 6 else 170
    if sum_low <= s <= sum_high:
        score -= 0.7
    elif s >= sum_tail:
        score += 0.6

    consec = sum(1 for i in range(n_pick - 1) if nums[i + 1] == nums[i] + 1)
    if consec >= 1:
        score += 0.5
    if consec >= 2:
        score += 0.4

    tails = len(set(n % 10 for n in nums))
    score += (tails - 4) * 0.2

    if hi in nums:
        score += 0.3

    return score


# -- モデル構築 --

def _build_model(draws, loto, params_map=None, selection_mode="coverage", portfolio_map=None):
    params_map = _params_map(params_map)
    portfolio_map = _portfolio_map(portfolio_map)
    cache_key = _cache_key(loto, params_map, selection_mode, portfolio_map)
    hist = draws[: min(params_map[loto]["lookback"], len(draws))]
    if cache_key in _LAST_MODEL_CACHE:
        cached = _LAST_MODEL_CACHE[cache_key]
        if cached.get("history_id") == tuple((d.number for d in hist)):
            return cached

    cfg = LOTO_CONFIG[loto]
    params = params_map[loto]
    lo, hi = cfg["range"]
    pick = cfg["pick"]
    hist = draws[: min(params["lookback"], len(draws))]
    if not hist:
        raise ValueError("history is empty")
    recent_len = min(params["recent"], len(hist))
    latest = hist[0].main
    adj = _adjacent_from_numbers(latest, cfg["range"])

    full_count = Counter()
    recent_count = Counter()
    decay_count = Counter()
    pair_recent = Counter()
    triple_recent = Counter()
    positions = {n: [] for n in range(lo, hi + 1)}

    for idx, d in enumerate(hist):
        full_count.update(d.main)
        if idx < recent_len:
            recent_count.update(d.main)
        decay_w = params["decay_base"] ** idx
        pair_w = params["pair_decay"] ** idx
        triple_w = params["triple_decay"] ** idx
        for n in d.main:
            decay_count[n] += decay_w
            positions[n].append(idx)
        for a, b in combinations(d.main, 2):
            pair_recent[_pair_key(a, b)] += pair_w
        if params["use_triple"]:
            for tri in combinations(d.main, 3):
                triple_recent[tri] += triple_w

    raw = {}
    exp_default = (hi - lo + 1) / pick
    for n in range(lo, hi + 1):
        pos = positions[n]
        if pos:
            gap = pos[0]
            intervals = [pos[i] - pos[i - 1] for i in range(1, len(pos))]
            avg_interval = _mean(intervals) if intervals else None
            med_interval = _median(intervals) if intervals else None
        else:
            gap = len(hist)
            avg_interval = None
            med_interval = None

        exp_gap = (len(hist) / max(1, full_count[n])) if full_count[n] else exp_default
        target_gap = med_interval if med_interval is not None else avg_interval if avg_interval is not None else exp_gap
        due = 1.0 - abs(gap - target_gap) / (target_gap + 1.0)
        overdue = gap / (exp_gap + 1.0)
        bridge = sum(pair_recent[_pair_key(n, m)] for m in latest if m != n)
        if pos and len(pos) >= 2:
            intervals = [pos[i] - pos[i - 1] for i in range(1, len(pos))]
            avg_int = _mean(intervals) or exp_gap
            cycle = 1.0 - abs(gap - avg_int) / (avg_int + 1.0)
        else:
            cycle = 0.0

        raw[n] = {
            "full": full_count[n],
            "recent": recent_count[n],
            "decay": decay_count[n],
            "gap": gap,
            "due": due,
            "overdue": overdue,
            "carry": 1 if n in latest else 0,
            "adj": 1 if n in adj else 0,
            "bridge": bridge,
            "cycle": cycle,
        }

    feature_names = list(next(iter(raw.values())).keys())
    means = {name: _mean([raw[n][name] for n in raw]) for name in feature_names}
    stdevs = {name: (_pstdev([raw[n][name] for n in raw]) or 1.0) for name in feature_names}

    composite = {}
    for n in raw:
        z = {name: (raw[n][name] - means[name]) / stdevs[name] for name in feature_names}
        composite[n] = sum(params["num_weights"].get(name, 0.0) * z[name] for name in feature_names)

    scaled = _scaled_scores(composite)
    ranked = sorted(composite, key=lambda n: (composite[n], raw[n]["recent"], raw[n]["full"], -n), reverse=True)
    hot_n = max(1, int(round((hi - lo + 1) * params["hot_pct"])))
    cold_n = max(1, int(round((hi - lo + 1) * params["cold_pct"])))
    hot = set(ranked[:hot_n])
    cold = set(ranked[-cold_n:])
    stats = {
        n: NumberStats(
            scaled[n],
            "HOT" if n in hot else "COLD" if n in cold else "MID",
        )
        for n in range(lo, hi + 1)
    }

    sums = [sum(d.main) for d in hist]
    odds = [sum(1 for n in d.main if n % 2 == 1) for d in hist]
    lows = [sum(1 for n in d.main if n <= cfg["small_max"]) for d in hist]
    consecs = [sum(1 for a, b in zip(d.main, d.main[1:]) if b == a + 1) for d in hist]
    spans = [d.main[-1] - d.main[0] for d in hist]
    tail_dups = [max(Counter(n % 10 for n in d.main).values()) for d in hist]

    repeats = []
    adj_hits = []
    for i in range(len(hist) - 1):
        prev = hist[i + 1].main
        curr = hist[i].main
        repeats.append(len(set(prev) & set(curr)))
        prev_adj = _adjacent_from_numbers(prev, cfg["range"])
        adj_hits.append(len(set(curr) & prev_adj))

    dist = {
        "sum_mean": _mean(sums),
        "sum_sd": _pstdev(sums) or 1.0,
        "odd_mode": _safe_mode(odds, cfg["odd_even_base"][0]),
        "low_mode": _safe_mode(lows, cfg["pick"] // 2),
        "consec_mode": _safe_mode(consecs, 1),
        "repeat_mean": _mean(repeats),
        "adj_mean": _mean(adj_hits),
        "span_mean": _mean(spans),
        "span_sd": _pstdev(spans) or 1.0,
        "tail_mode": _safe_mode(tail_dups, 2),
    }

    comp_min = min(composite.values())
    comp_max = max(composite.values())
    comp_den = comp_max - comp_min or 1.0
    num_value = {
        n: 0.55 + 0.90 * (composite[n] - comp_min) / comp_den + 0.10 * max(0.0, raw[n]["due"])
        for n in range(lo, hi + 1)
    }
    pair_vals_all = list(pair_recent.values()) or [0.0]
    pair_min, pair_max = min(pair_vals_all), max(pair_vals_all)
    pair_den = pair_max - pair_min or 1.0
    pair_value = {
        k: (pair_recent[k] - pair_min) / pair_den
        for k in pair_recent
    }

    thirds = (hi - lo + 1) // 3
    band_map = {}
    for n in range(lo, hi + 1):
        if n <= lo + thirds - 1:
            band_map[n] = "L"
        elif n <= lo + 2 * thirds - 1:
            band_map[n] = "M"
        else:
            band_map[n] = "H"

    model = {
        "loto": loto,
        "history": hist,
        "history_id": tuple((d.number for d in hist)),
        "stats": stats,
        "composite": composite,
        "raw": raw,
        "pair_recent": pair_recent,
        "triple_recent": triple_recent,
        "adj": adj,
        "latest": latest,
        "dist": dist,
        "ranked": ranked,
        "num_value": num_value,
        "pair_value": pair_value,
        "band_map": band_map,
        "params_map": params_map,
        "portfolio_map": portfolio_map,
        "selection_mode": selection_mode,
    }
    _LAST_MODEL_CACHE[cache_key] = model
    return model


# -- 分析 --

def _cold_range(num_sets, loto, params_map=None):
    params_map = _params_map(params_map)
    params = params_map[loto]
    pick = LOTO_CONFIG[loto]["pick"]
    expected = num_sets * pick * params["cold_pct"]
    upper = max(1, int(round(expected + 1)))
    return 0, upper


def calc_frequency(draws, num_range):
    loto = _loto_from_range(num_range)
    return _build_model(draws, loto)["stats"]


def recent_trend(draws, config):
    r5, r3 = draws[:5], draws[:3]
    sums = [sum(d.main) for d in r5]
    odds = [sum(1 for n in d.main if n % 2 == 1) for d in r5]
    smalls = [sum(1 for n in d.main if n <= config["small_max"]) for d in r5]
    tc = Counter()
    for d in r3:
        tc.update(n % 10 for n in d.main)
    return {
        "sums": sums,
        "sum_avg": _mean(sums),
        "odd_ratios": odds,
        "odd_avg": _mean(odds),
        "small_counts": smalls,
        "small_avg": _mean(smalls),
        "top_tails": tc.most_common(5),
    }


def adjacent_numbers(draws, num_range):
    return _adjacent_from_numbers(draws[0].main, num_range)


# -- 生成 --

def has_triple_consecutive(nums):
    s = sorted(nums)
    return any(s[i + 1] == s[i] + 1 and s[i + 2] == s[i] + 2 for i in range(len(s) - 2))


def validate_set(nums, config, stats, adj, set_type=None, params_map=None):
    loto = _loto_from_range(config["range"])
    params_map = _params_map(params_map)
    hot_limit = params_map[loto]["hot_limit"]
    if len(nums) != config["pick"] or len(set(nums)) != config["pick"]:
        return False
    if any(n < config["range"][0] or n > config["range"][1] for n in nums):
        return False
    if has_triple_consecutive(nums):
        return False
    odd_count = sum(1 for n in nums if n % 2 == 1)
    if abs(odd_count - config["odd_even_base"][0]) > 1:
        return False
    hot_count = sum(1 for n in nums if stats[n].category == "HOT")
    if hot_count > hot_limit:
        return False
    return True


def _candidate_pool(model, loto, params_map=None):
    params_map = _params_map(params_map)
    params = params_map[loto]
    cfg = LOTO_CONFIG[loto]
    lo, hi = cfg["range"]
    ranked = model["ranked"]
    pool = ranked[: params["top_m"]]
    overdue_sorted = sorted(range(lo, hi + 1), key=lambda n: model["raw"][n]["overdue"], reverse=True)
    for n in overdue_sorted[: params["extra_overdue"]]:
        if n not in pool and len(pool) < params["max_pool"]:
            pool.append(n)
    for n in sorted(model["adj"]):
        if n not in pool and len(pool) < params["max_pool"]:
            pool.append(n)
    if params["extra_cold"]:
        cold_sorted = sorted(
            [n for n in range(lo, hi + 1) if model["stats"][n].category == "COLD"],
            key=lambda n: model["composite"][n],
            reverse=True,
        )
        for n in cold_sorted[: params["extra_cold"]]:
            if n not in pool and len(pool) < params["max_pool"]:
                pool.append(n)
    return sorted(pool)


def _balanced_band_pool(model, loto, size):
    cfg = LOTO_CONFIG[loto]
    lo, hi = cfg["range"]
    thirds = (hi - lo + 1) // 3
    params = model["params_map"][loto]
    ranked = model["ranked"]
    low = [n for n in ranked if n <= lo + thirds - 1]
    mid = [n for n in ranked if lo + thirds <= n <= lo + 2 * thirds - 1]
    high = [n for n in ranked if n > lo + 2 * thirds - 1]
    buckets = [low, mid, high]
    quotas = [size // 3, size // 3, size - 2 * (size // 3)]
    pool = []
    for bucket, quota in zip(buckets, quotas):
        for n in bucket:
            if n not in pool:
                pool.append(n)
            if sum(1 for x in pool if x in bucket) >= quota:
                break
    for n in ranked:
        if n not in pool and len(pool) < size:
            pool.append(n)
    return sorted(pool[:size])


def _replace_tail(core, extras, size, replace_count):
    core = list(core[:size])
    k = max(0, min(len(core), replace_count))
    keep = core[: max(0, size - k)]
    out = list(keep)
    for n in extras:
        if n not in out:
            out.append(n)
        if len(out) >= size:
            break
    return sorted(out[:size])


def _coverage_pools(model, loto, params_map=None, portfolio_map=None):
    params_map = _params_map(params_map)
    portfolio_map = _portfolio_map(portfolio_map)
    params = params_map[loto]
    cfg = LOTO_CONFIG[loto]
    lo, hi = cfg["range"]
    ranked = model["ranked"]
    port = portfolio_map[loto]
    size = port["pool_size"]
    core = ranked[: max(size, params["top_m"]) ]
    overdue_sorted = sorted(range(lo, hi + 1), key=lambda n: model["raw"][n]["overdue"], reverse=True)
    cold_sorted = sorted(
        [n for n in range(lo, hi + 1) if model["stats"][n].category == "COLD"],
        key=lambda n: model["composite"][n],
        reverse=True,
    )
    adj_sorted = sorted(model["adj"], key=lambda n: model["composite"].get(n, -999), reverse=True)
    thirds = (hi - lo + 1) // 3
    low_ranked = [n for n in ranked if n <= lo + thirds - 1]
    mid_ranked = [n for n in ranked if lo + thirds <= n <= lo + 2 * thirds - 1]
    high_ranked = [n for n in ranked if n > lo + 2 * thirds - 1]
    nohot_ranked = [n for n in ranked if model["stats"][n].category != "HOT"]

    pools = []
    pools.append(sorted(core[:size]))
    pools.append(_replace_tail(core, overdue_sorted + adj_sorted, size, 4))
    pools.append(_replace_tail(core, adj_sorted + overdue_sorted, size, 4))
    pools.append(_balanced_band_pool(model, loto, size))
    pools.append(_replace_tail(core, low_ranked + mid_ranked, size, 5))
    pools.append(_replace_tail(core, high_ranked + mid_ranked, size, 5))
    pools.append(_replace_tail(core, nohot_ranked + cold_sorted + overdue_sorted, size, 4))

    uniq = []
    seen = set()
    for p in pools:
        key = tuple(p)
        if len(p) >= cfg["pick"] and key not in seen:
            uniq.append(p)
            seen.add(key)
    return uniq


def _candidate_metrics(nums, model, loto):
    cfg = LOTO_CONFIG[loto]
    pair_recent = model["pair_recent"]
    triple_recent = model["triple_recent"]
    raw = model["raw"]
    dist = model["dist"]
    stats = model["stats"]
    pair_vals = [pair_recent[_pair_key(a, b)] for a, b in combinations(nums, 2)]
    if model["params_map"][loto]["use_triple"]:
        triple_vals = [triple_recent[tri] for tri in combinations(nums, 3)]
    else:
        triple_vals = []
    odd = sum(1 for n in nums if n % 2 == 1)
    low = sum(1 for n in nums if n <= cfg["small_max"])
    consec = sum(1 for a, b in zip(nums, nums[1:]) if b == a + 1)
    repeat = len(set(nums) & set(model["latest"]))
    adj_hits = len(set(nums) & model["adj"])
    hot_count = sum(1 for n in nums if stats[n].category == "HOT")
    cold_count = sum(1 for n in nums if stats[n].category == "COLD")
    overdue_avg = _mean([raw[n]["overdue"] for n in nums])
    num_sum = sum(model["composite"][n] for n in nums)
    tail_dup = max(Counter(n % 10 for n in nums).values())
    nums_sum = sum(nums)
    span = nums[-1] - nums[0]
    return {
        "nums": nums,
        "num_sum": num_sum,
        "pair_mean": _mean(pair_vals),
        "pair_max": max(pair_vals) if pair_vals else 0.0,
        "triple_mean": _mean(triple_vals) if triple_vals else 0.0,
        "triple_max": max(triple_vals) if triple_vals else 0.0,
        "sum": nums_sum,
        "odd": odd,
        "low": low,
        "consec": consec,
        "repeat": repeat,
        "adj_hits": adj_hits,
        "span": span,
        "hot_count": hot_count,
        "cold_count": cold_count,
        "overdue_avg": overdue_avg,
        "sum_dist": abs(nums_sum - dist["sum_mean"]),
        "cycle_mean": _mean([raw[n]["cycle"] for n in nums]),
        "tail_dup": tail_dup,
        "pair_keys": tuple(_pair_key(a, b) for a, b in combinations(nums, 2)),
        "band_counts": Counter(model["band_map"][n] for n in nums),
        "sum_bin": _zbin(nums_sum, dist["sum_mean"], dist["sum_sd"]),
        "span_bin": _zbin(span, dist["span_mean"], dist["span_sd"]),
        "low_bin": low,
    }


def _score_candidate(metrics, loto, params_map=None, model=None):
    params_map = _params_map(params_map)
    weights = params_map[loto]["set_weights"]
    dist = model["dist"] if model is not None else _LAST_MODEL_CACHE[loto]["dist"]
    score = metrics["num_sum"]
    score += weights["pair_mean"] * metrics["pair_mean"]
    score += weights["pair_max"] * metrics["pair_max"]
    score += weights["triple_mean"] * metrics["triple_mean"]
    score += weights["triple_max"] * metrics["triple_max"]
    score += weights["consec"] * (-abs(metrics["consec"] - dist["consec_mode"]))
    score += weights["odd"] * (-abs(metrics["odd"] - dist["odd_mode"]))
    score += weights["low"] * (-abs(metrics["low"] - dist["low_mode"]))
    score += weights["sum"] * (-metrics["sum_dist"] / (dist["sum_sd"] or 1.0))
    score += weights["repeat"] * (-abs(metrics["repeat"] - dist["repeat_mean"]))
    score += weights["adj"] * (-abs(metrics["adj_hits"] - dist["adj_mean"]))
    score += weights["span"] * (-abs(metrics["span"] - dist["span_mean"]) / (dist["span_sd"] or 1.0))
    score += weights["cycle"] * metrics["cycle_mean"]
    score += weights["tail"] * (-abs(metrics["tail_dup"] - dist["tail_mode"]))
    return score


def _enumerate_candidates(model, loto, selection_mode="coverage", params_map=None, portfolio_map=None):
    params_map = _params_map(params_map)
    portfolio_map = _portfolio_map(portfolio_map)
    cfg = LOTO_CONFIG[loto]
    params = params_map[loto]
    stats = model["stats"]
    pools = [_candidate_pool(model, loto, params_map)] if selection_mode == "greedy" else _coverage_pools(model, loto, params_map, portfolio_map)

    all_candidates = []
    seen = set()
    for pool in pools:
        local = []
        for nums in combinations(pool, cfg["pick"]):
            nums = tuple(sorted(nums))
            if nums in seen:
                continue
            if not validate_set(nums, cfg, stats, model["adj"], params_map=params_map):
                continue
            metrics = _candidate_metrics(nums, model, loto)
            metrics["score"] = _score_candidate(metrics, loto, params_map=params_map, model=model)
            local.append(metrics)
            seen.add(nums)
        local.sort(key=lambda m: (m["score"], m["pair_mean"], m["num_sum"], -m["sum_dist"]), reverse=True)
        all_candidates.extend(local[: params.get("pool_keep_per_pool", 500)])

    all_candidates.sort(key=lambda m: (m["score"], m["pair_mean"], m["num_sum"], -m["sum_dist"]), reverse=True)
    # dedupe after slicing pools
    dedup = []
    used = set()
    for m in all_candidates:
        if m["nums"] not in used:
            dedup.append(m)
            used.add(m["nums"])
    keep = portfolio_map[loto].get("candidate_keep", 1200) if selection_mode == "coverage" else 2000
    return dedup[:keep]


def _normalize_candidate_scores(candidates):
    if not candidates:
        return
    vals = [c["score"] for c in candidates]
    mn, mx = min(vals), max(vals)
    den = mx - mn or 1.0
    for c in candidates:
        c["score_norm"] = (c["score"] - mn) / den


def _selected_valid(selected_metrics, loto, params_map=None, num_sets=None):
    params_map = _params_map(params_map)
    max_overlap = params_map[loto]["max_overlap"]
    if num_sets is not None and len(selected_metrics) != num_sets:
        return False
    return all(
        len(set(a["nums"]) & set(b["nums"])) <= max_overlap
        for a, b in combinations(selected_metrics, 2)
    )


def _make_portfolio_state(selected, loto, model, port):
    state = {
        "selected": list(selected),
        "num_counts": Counter(),
        "pair_covered": set(),
        "low_bins": set(),
        "sum_bins": set(),
        "span_bins": set(),
        "indiv_sum": 0.0,
        "score": 0.0,
    }
    for cand in selected:
        state["indiv_sum"] += cand.get("score_norm", 0.0)
        for n in cand["nums"]:
            state["num_counts"][n] += 1
        state["pair_covered"].update(cand["pair_keys"])
        state["low_bins"].add(cand["low_bin"])
        state["sum_bins"].add(cand["sum_bin"])
        state["span_bins"].add(cand["span_bin"])
    state["score"] = _portfolio_score_state(state, loto, port, model)
    return state


def _state_with_candidate(state, cand, loto, model, port):
    new_state = {
        "selected": state["selected"] + [cand],
        "num_counts": state["num_counts"].copy(),
        "pair_covered": set(state["pair_covered"]),
        "low_bins": set(state["low_bins"]),
        "sum_bins": set(state["sum_bins"]),
        "span_bins": set(state["span_bins"]),
        "indiv_sum": state["indiv_sum"] + cand.get("score_norm", 0.0),
        "score": 0.0,
    }
    for n in cand["nums"]:
        new_state["num_counts"][n] += 1
    new_state["pair_covered"].update(cand["pair_keys"])
    new_state["low_bins"].add(cand["low_bin"])
    new_state["sum_bins"].add(cand["sum_bin"])
    new_state["span_bins"].add(cand["span_bin"])
    new_state["score"] = _portfolio_score_state(new_state, loto, port, model)
    return new_state


def _search_score_complete(candidates, loto, num_sets, params_map=None, seed_selected=None):
    params_map = _params_map(params_map)
    max_overlap = params_map[loto]["max_overlap"]
    seed_selected = list(seed_selected or [])
    _normalize_candidate_scores(candidates)

    filtered = [c for c in candidates if c not in seed_selected]
    nums_sets = [set(c["nums"]) for c in filtered]
    seed_sets = [set(c["nums"]) for c in seed_selected]

    base_indices = []
    for idx, cand in enumerate(filtered):
        nums = nums_sets[idx]
        if all(len(nums & prev) <= max_overlap for prev in seed_sets):
            base_indices.append(idx)
    base_indices.sort(
        key=lambda i: (
            filtered[i].get("score_norm", 0.0),
            filtered[i]["pair_mean"],
            filtered[i]["num_sum"],
            -filtered[i]["sum_dist"],
        ),
        reverse=True,
    )

    best_partial = list(seed_selected)
    branch_limits = [320, 240, 180, 120, 80]
    node_budget = max(120000, len(base_indices) * 90)
    nodes = [0]

    def dfs(selected_indices, avail_indices, depth=0):
        nodes[0] += 1
        current_selected = seed_selected + [filtered[i] for i in selected_indices]
        if len(current_selected) > len(best_partial):
            best_partial[:] = current_selected
        if len(current_selected) == num_sets:
            return current_selected
        if nodes[0] > node_budget:
            return None
        need = num_sets - len(current_selected)
        if len(avail_indices) < need:
            return None

        av_sorted = sorted(
            avail_indices,
            key=lambda i: (
                filtered[i].get("score_norm", 0.0),
                filtered[i]["pair_mean"],
                filtered[i]["num_sum"],
                -filtered[i]["sum_dist"],
            ),
            reverse=True,
        )
        limit = branch_limits[min(depth, len(branch_limits) - 1)]
        for idx in av_sorted[:limit]:
            nums = nums_sets[idx]
            ok = True
            for prev_idx in selected_indices:
                if len(nums & nums_sets[prev_idx]) > max_overlap:
                    ok = False
                    break
            if not ok:
                continue
            next_avail = []
            for j in av_sorted:
                if j <= idx:
                    continue
                nums_j = nums_sets[j]
                if len(nums & nums_j) > max_overlap:
                    continue
                if any(len(nums_j & nums_sets[prev_idx]) > max_overlap for prev_idx in selected_indices):
                    continue
                next_avail.append(j)
            result = dfs(selected_indices + [idx], next_avail, depth + 1)
            if result is not None:
                return result
        return None

    found = dfs([], base_indices, 0)
    selected = found if found is not None else best_partial
    fully_satisfied = len(selected) >= num_sets and _selected_valid(selected[:num_sets], loto, params_map=params_map, num_sets=num_sets)
    return selected[:num_sets], fully_satisfied


def _select_greedy(candidates, loto, num_sets, params_map=None):
    return _search_score_complete(candidates, loto, num_sets, params_map=params_map)



def _portfolio_score_state(state, loto, port, model):
    # state: dict with selected, num_counts, pair_covered, low_bins, sum_bins, span_bins, indiv_sum
    num_counts = state["num_counts"]
    unique_nums = [n for n, c in num_counts.items() if c >= 1]
    weighted_unique = sum(model["num_value"][n] for n in unique_nums)
    repeat_pen = 0.0
    hard_pen = 0.0
    for n, c in num_counts.items():
        if c >= 2:
            repeat_pen += (c - 1) * model["num_value"][n]
        if c >= 3:
            hard_pen += ((c - 2) ** 2) * model["num_value"][n]
    pair_bonus = sum(model["pair_value"].get(p, 0.0) for p in state["pair_covered"])
    score = port["indiv"] * state["indiv_sum"]
    score += port["new_num"] * weighted_unique
    score += port["new_pair"] * pair_bonus
    score -= port["num_repeat"] * repeat_pen
    score -= port["num_repeat_hard"] * hard_pen
    score += port["new_lowbin"] * len(state["low_bins"])
    score += port["new_sumbin"] * len(state["sum_bins"])
    score += port["new_spanbin"] * len(state["span_bins"])
    unique_target = port["min_unique_target"]
    if len(unique_nums) >= unique_target:
        score += port["min_unique_bonus"] * (len(unique_nums) - unique_target + 1)
    return score

def _coverage_retry_maps(model, loto, params_map=None, portfolio_map=None):
    params_map = deepcopy(_params_map(params_map))
    portfolio_map = deepcopy(_portfolio_map(portfolio_map))
    pool_count = len(_coverage_pools(model, loto, params_map=params_map, portfolio_map=portfolio_map))
    params = params_map[loto]
    port = portfolio_map[loto]
    port["candidate_keep"] = min(
        max(port["candidate_keep"] * 3, port["candidate_keep"] + 1200),
        params["pool_keep_per_pool"] * pool_count,
    )
    port["beam_width"] = max(port["beam_width"], 24 if loto == "loto6" else 28)
    port["expand_per_state"] = max(port["expand_per_state"], 180 if loto == "loto6" else 220)
    return params_map, portfolio_map


def _search_coverage_complete(candidates, loto, num_sets, model, params_map=None, portfolio_map=None, seed_selected=None):
    params_map = _params_map(params_map)
    portfolio_map = _portfolio_map(portfolio_map)
    port = portfolio_map[loto]
    max_overlap = params_map[loto]["max_overlap"]
    seed_selected = list(seed_selected or [])
    _normalize_candidate_scores(candidates)

    filtered = [c for c in candidates if c not in seed_selected]
    nums_sets = [set(c["nums"]) for c in filtered]
    seed_sets = [set(c["nums"]) for c in seed_selected]
    base_indices = []
    for idx, cand in enumerate(filtered):
        nums = nums_sets[idx]
        if all(len(nums & prev) <= max_overlap for prev in seed_sets):
            base_indices.append(idx)
    base_indices.sort(
        key=lambda i: (
            filtered[i].get("score_norm", 0.0),
            filtered[i]["pair_mean"],
            -filtered[i]["hot_count"],
            filtered[i]["span"],
            -filtered[i]["sum_dist"],
        ),
        reverse=True,
    )

    best_partial = _make_portfolio_state(seed_selected, loto, model, port)
    best_full = None
    branch_limits = [160, 120, 80, 60, 40]
    node_budget = max(60000, len(base_indices) * 18)
    nodes = [0]

    def dfs(selected_indices, avail_indices, state, depth=0):
        nonlocal best_full, best_partial
        nodes[0] += 1
        if len(state["selected"]) > len(best_partial["selected"]) or (
            len(state["selected"]) == len(best_partial["selected"]) and state["score"] > best_partial["score"]
        ):
            best_partial = state
        if len(state["selected"]) == num_sets:
            if best_full is None or state["score"] > best_full["score"]:
                best_full = state
            return
        if nodes[0] > node_budget:
            return
        need = num_sets - len(state["selected"])
        if len(avail_indices) < need:
            return

        scored = []
        for idx in avail_indices:
            nums = nums_sets[idx]
            if any(len(nums & set(prev["nums"])) > max_overlap for prev in state["selected"]):
                continue
            cand = filtered[idx]
            new_state = _state_with_candidate(state, cand, loto, model, port)
            scored.append((
                new_state["score"],
                cand.get("score_norm", 0.0),
                cand["pair_mean"],
                idx,
                new_state,
            ))
        if len(scored) < need:
            return
        scored.sort(reverse=True)
        limit = branch_limits[min(depth, len(branch_limits) - 1)]
        for _, __, ___, idx, new_state in scored[:limit]:
            nums = nums_sets[idx]
            next_avail = []
            for _, __, ___, j, _dummy in scored:
                if j <= idx:
                    continue
                if len(nums & nums_sets[j]) > max_overlap:
                    continue
                next_avail.append(j)
            dfs(selected_indices + [idx], next_avail, new_state, depth + 1)

    dfs([], base_indices, _make_portfolio_state(seed_selected, loto, model, port), 0)
    if best_full is not None and _selected_valid(best_full["selected"], loto, params_map=params_map, num_sets=num_sets):
        return best_full["selected"][:num_sets], True
    return best_partial["selected"][:num_sets], False


def _select_coverage(candidates, loto, num_sets, model, params_map=None, portfolio_map=None):
    params_map = _params_map(params_map)
    portfolio_map = _portfolio_map(portfolio_map)
    port = portfolio_map[loto]
    max_overlap = params_map[loto]["max_overlap"]
    _normalize_candidate_scores(candidates)

    candidates = sorted(
        candidates,
        key=lambda m: (
            m.get("score_norm", 0.0),
            m["pair_mean"],
            -m["hot_count"],
            m["span"],
            -m["sum_dist"],
        ),
        reverse=True,
    )[: port["candidate_keep"]]

    empty = {
        "selected": [],
        "num_counts": Counter(),
        "pair_covered": set(),
        "low_bins": set(),
        "sum_bins": set(),
        "span_bins": set(),
        "indiv_sum": 0.0,
        "score": 0.0,
        "fully": True,
    }
    beam = [empty]

    for depth in range(num_sets):
        next_states = []
        for state in beam:
            tried = 0
            for cand in candidates:
                if cand in state["selected"]:
                    continue
                overlaps = [len(set(cand["nums"]) & set(prev["nums"])) for prev in state["selected"]]
                ov = max(overlaps) if overlaps else 0
                if ov > max_overlap:
                    continue
                new_state = {
                    "selected": state["selected"] + [cand],
                    "num_counts": state["num_counts"].copy(),
                    "pair_covered": set(state["pair_covered"]),
                    "low_bins": set(state["low_bins"]),
                    "sum_bins": set(state["sum_bins"]),
                    "span_bins": set(state["span_bins"]),
                    "indiv_sum": state["indiv_sum"] + cand.get("score_norm", 0.0),
                    "fully": state["fully"],
                }
                for n in cand["nums"]:
                    new_state["num_counts"][n] += 1
                new_state["pair_covered"].update(cand["pair_keys"])
                new_state["low_bins"].add(cand["low_bin"])
                new_state["sum_bins"].add(cand["sum_bin"])
                new_state["span_bins"].add(cand["span_bin"])
                new_state["score"] = _portfolio_score_state(new_state, loto, port, model)
                next_states.append(new_state)
                tried += 1
                if tried >= port["expand_per_state"]:
                    break
        if not next_states:
            break
        next_states.sort(
            key=lambda s: (
                len(s["selected"]) == num_sets and _selected_valid(s["selected"], loto, params_map=params_map, num_sets=num_sets),
                len(s["selected"]),
                s["score"],
                len([n for n, c in s["num_counts"].items() if c >= 1]),
                len(s["pair_covered"]),
                -sum(max(0, c - 1) for c in s["num_counts"].values()),
            ),
            reverse=True,
        )
        pruned = []
        seen_sig = set()
        for st in next_states:
            sig = (
                tuple(sorted(m["nums"] for m in st["selected"])),
                tuple(sorted((n, c) for n, c in st["num_counts"].items() if c >= 2)),
            )
            if sig in seen_sig:
                continue
            pruned.append(st)
            seen_sig.add(sig)
            if len(pruned) >= port["beam_width"]:
                break
        beam = pruned

    best = max(
        beam or [empty],
        key=lambda s: (
            len(s["selected"]) == num_sets and _selected_valid(s["selected"], loto, params_map=params_map, num_sets=num_sets),
            len(s["selected"]),
            s["score"],
            len([n for n, c in s["num_counts"].items() if c >= 1]),
        ),
    )
    selected = best["selected"][:num_sets]
    if len(selected) == num_sets and _selected_valid(selected, loto, params_map=params_map, num_sets=num_sets):
        return selected, True

    retry_params_map, retry_portfolio_map = _coverage_retry_maps(model, loto, params_map=params_map, portfolio_map=portfolio_map)
    retry_candidates = _enumerate_candidates(model, loto, selection_mode="coverage", params_map=retry_params_map, portfolio_map=retry_portfolio_map)
    retry_selected, retry_full = _search_coverage_complete(
        retry_candidates,
        loto,
        num_sets,
        model,
        params_map=retry_params_map,
        portfolio_map=retry_portfolio_map,
    )
    if retry_full and _selected_valid(retry_selected, loto, params_map=retry_params_map, num_sets=num_sets):
        return retry_selected, True

    score_selected, score_full = _search_score_complete(retry_candidates, loto, num_sets, params_map=retry_params_map)
    return score_selected, bool(score_full)


def _assign_labels(selected_metrics, num_sets):
    if not selected_metrics:
        return []
    remaining = list(range(len(selected_metrics)))
    assigned = {}

    reverse_idx = max(
        remaining,
        key=lambda i: (
            selected_metrics[i]["cold_count"],
            selected_metrics[i]["overdue_avg"],
            -selected_metrics[i]["num_sum"],
        ),
    )
    assigned[reverse_idx] = "逆張り"
    remaining.remove(reverse_idx)

    if remaining:
        king_a = max(
            remaining,
            key=lambda i: (
                selected_metrics[i]["pair_mean"],
                selected_metrics[i]["num_sum"],
                -selected_metrics[i]["sum_dist"],
            ),
        )
        assigned[king_a] = "王道A"
        remaining.remove(king_a)

    if remaining:
        spread_a = max(
            remaining,
            key=lambda i: (
                selected_metrics[i]["span"],
                len(set(selected_metrics[i]["nums"])),
                selected_metrics[i]["adj_hits"],
            ),
        )
        assigned[spread_a] = "分散A"
        remaining.remove(spread_a)

    if remaining:
        king_b = max(remaining, key=lambda i: (selected_metrics[i]["num_sum"], selected_metrics[i]["pair_max"]))
        assigned[king_b] = "王道B"
        remaining.remove(king_b)

    for i in remaining:
        assigned[i] = "分散B"

    return [(assigned[i], selected_metrics[i]["nums"]) for i in range(len(selected_metrics))]


def _enumerate_ev_candidates(model, loto, num_samples=15000, seed=0):
    # Bypass statistical pool restrictions; sample broadly from the full number
    # space and rank by EV unpopularity only. HOT/overdue/pair features are
    # deliberately ignored — they are not predictive for independent draws.
    cfg = LOTO_CONFIG[loto]
    lo, hi = cfg["range"]
    pick = cfg["pick"]
    rng = random.Random(seed)
    pool = list(range(lo, hi + 1))

    seen = set()
    candidates = []
    attempts = 0
    max_attempts = num_samples * 4
    while len(candidates) < num_samples and attempts < max_attempts:
        attempts += 1
        nums = tuple(sorted(rng.sample(pool, pick)))
        if nums in seen:
            continue
        seen.add(nums)
        odd = sum(1 for n in nums if n % 2 == 1)
        if odd == 0 or odd == pick:
            continue
        if has_triple_consecutive(nums):
            continue
        metrics = _candidate_metrics(nums, model, loto)
        ev = _ev_unpopularity(nums, cfg)
        metrics["ev_score"] = ev
        metrics["score"] = ev
        candidates.append(metrics)

    candidates.sort(key=lambda m: (m["ev_score"], m["sum"]), reverse=True)
    return candidates


def _select_ev_portfolio(candidates, loto, num_sets, max_overlap=3):
    # Greedy top-EV selection with a light diversity cap so the 5 sets do not
    # collapse into the same few numbers.
    chosen = []
    for c in candidates:
        if len(chosen) >= num_sets:
            break
        nums_set = set(c["nums"])
        if any(len(nums_set & set(s["nums"])) > max_overlap for s in chosen):
            continue
        chosen.append(c)
    if len(chosen) < num_sets:
        for c in candidates:
            if c in chosen:
                continue
            chosen.append(c)
            if len(chosen) >= num_sets:
                break
    return chosen[:num_sets]


def _generate_result_from_model(model, loto, num_sets=5, selection_mode="coverage", params_map=None, portfolio_map=None, ev_mode=True):
    params_map = _params_map(params_map)
    portfolio_map = _portfolio_map(portfolio_map)

    if ev_mode:
        candidates = _enumerate_ev_candidates(model, loto)
        if not candidates:
            raise RuntimeError("EV候補生成失敗")
        selected = _select_ev_portfolio(candidates, loto, num_sets)
        labeled = _assign_labels(selected, num_sets)
        if len(labeled) < num_sets:
            raise RuntimeError("EV選択失敗")
        # Honest fully_satisfied: check actual overlap constraint instead of
        # always returning True. _select_ev_portfolio has a fallback path that
        # can violate max_overlap when candidates are exhausted.
        ev_max_overlap = 3
        fully = all(
            len(set(a["nums"]) & set(b["nums"])) <= ev_max_overlap
            for a, b in combinations(selected, 2)
        )
        return GenerateResult(labeled, fully), selected

    candidates = _enumerate_candidates(model, loto, selection_mode=selection_mode, params_map=params_map, portfolio_map=portfolio_map)
    if not candidates:
        raise RuntimeError("生成失敗")
    if selection_mode == "greedy":
        selected, fully_satisfied = _select_greedy(candidates, loto, num_sets, params_map=params_map)
        if len(selected) < num_sets or not fully_satisfied:
            retry_params_map, retry_portfolio_map = _coverage_retry_maps(model, loto, params_map=params_map, portfolio_map=portfolio_map)
            retry_candidates = _enumerate_candidates(model, loto, selection_mode="coverage", params_map=retry_params_map, portfolio_map=retry_portfolio_map)
            retry_selected, retry_full = _search_score_complete(retry_candidates, loto, num_sets, params_map=retry_params_map)
            if retry_full and len(retry_selected) >= num_sets:
                selected, fully_satisfied = retry_selected, True
    else:
        selected, fully_satisfied = _select_coverage(candidates, loto, num_sets, model, params_map=params_map, portfolio_map=portfolio_map)
    labeled = _assign_labels(selected, num_sets)
    if len(labeled) < num_sets:
        raise RuntimeError("生成失敗")
    return GenerateResult(labeled, fully_satisfied), selected


def generate_sets(config, stats, adj, trend, num_sets=5):
    # 既存互換用: 直前に作った最終モデルキャッシュを利用する
    loto = _loto_from_range(config["range"])
    model = _build_model(_LAST_MODEL_CACHE["__draws__"], loto)
    gen, _ = _generate_result_from_model(model, loto, num_sets=num_sets)
    return gen


def generate_from_draws(draws, loto, num_sets=5, params_map=None, selection_mode="coverage", portfolio_map=None, ev_mode=True):
    params_map = _params_map(params_map)
    portfolio_map = _portfolio_map(portfolio_map)
    model = _build_model(draws, loto, params_map=params_map, selection_mode=selection_mode, portfolio_map=portfolio_map)
    gen, selected_metrics = _generate_result_from_model(model, loto, num_sets=num_sets, selection_mode=selection_mode, params_map=params_map, portfolio_map=portfolio_map, ev_mode=ev_mode)
    return model, gen, selected_metrics


# -- メインAPI --

def run(draws, loto, num_sets=5, ev_mode=True):
    config = LOTO_CONFIG[loto]
    if len(draws) < 10:
        print(f"エラー: データ不足（{len(draws)}回）")
        return

    _LAST_MODEL_CACHE.clear()
    _LAST_MODEL_CACHE["__draws__"] = draws
    model, gen, selected_metrics = generate_from_draws(draws, loto, num_sets=num_sets, params_map=MODEL_PARAMS, selection_mode="coverage", portfolio_map=PORTFOLIO_PARAMS, ev_mode=ev_mode)
    stats = model["stats"]
    trend = recent_trend(draws, config)
    adj = model["adj"]
    lo, hi = config["range"]

    print(f"期間: 第{draws[-1].number}回〜第{draws[0].number}回（{len(draws)}回分）\n")

    hot = sorted([n for n, s in stats.items() if s.category == "HOT"], key=lambda n: stats[n].score, reverse=True)
    mid = sorted([n for n, s in stats.items() if s.category == "MID"], key=lambda n: stats[n].score, reverse=True)
    cold = sorted([n for n, s in stats.items() if s.category == "COLD"], key=lambda n: stats[n].score, reverse=True)
    print("【出現頻度スコア】")
    for rs in range(0, hi - lo + 1, 10):
        row = list(range(lo + rs, min(lo + rs + 10, hi + 1)))
        print("  番号:", " ".join(f"{n:4d}" for n in row))
        print("  Score:", " ".join(f"{stats[n].score:4d}" for n in row))
        print("  分類:", " ".join(f"{stats[n].category:>4}" for n in row))
        print()
    print(f"HOT ({len(hot)}): {hot}")
    print(f"MID ({len(mid)}): {mid}")
    print(f"COLD ({len(cold)}): {cold}\n")

    print("【直近傾向】")
    for d in draws[:5]:
        s = sum(d.main)
        odd = sum(1 for n in d.main if n % 2 == 1)
        print(f"  第{d.number}回: 合計={s:3d}  奇:偶={odd}:{len(d.main) - odd}  {d.main}")
    print(f"  合計平均: {trend['sum_avg']:.1f}")
    print(f"  頻出末尾: {', '.join(f'{t}({c}回)' for t, c in trend['top_tails'])}\n")

    print(f"【±1近接】直近: {draws[0].main}")
    print(f"  候補: {sorted(adj)}\n")

    ev_lookup = {}
    if ev_mode:
        for m in selected_metrics:
            ev_lookup[tuple(m["nums"])] = m.get("ev_score", 0.0)

    print("【生成結果】" + ("（EV最適化モード：被りにくい構成優先）" if ev_mode else ""))
    header_ev = f" | {'EV':>5}" if ev_mode else ""
    print(f"{'組':>2} | {'タイプ':>6} | {'数字':<30} | {'合計':>4} | {'奇:偶':>5} | {'HOT':>3} | {'COLD':>4}{header_ev}")
    print("-" * (72 + (8 if ev_mode else 0)))
    for i, (st, nums) in enumerate(gen.sets, 1):
        s = sum(nums)
        odd = sum(1 for n in nums if n % 2 == 1)
        h = sum(1 for n in nums if stats[n].category == "HOT")
        c = sum(1 for n in nums if stats[n].category == "COLD")
        ev_cell = f" | {ev_lookup.get(tuple(nums), 0.0):5.2f}" if ev_mode else ""
        print(f"{i:2d} | {st:>6} | {' '.join(f'{n:2d}' for n in nums):<30} | {s:4d} | {odd}:{len(nums) - odd:<3} | {h:3d} | {c:4d}{ev_cell}")
    print()

    if ev_mode:
        ht = config.get("high_threshold", 31)
        high_cnt = [sum(1 for n in ns if n > ht) for _, ns in gen.sets]
        date_only = sum(1 for _, ns in gen.sets if all(n <= 31 for n in ns))
        print(f"【EV要約】高位(>{ht})平均: {sum(high_cnt)/len(high_cnt):.1f}個/組  日付のみ構成: {date_only}/{len(gen.sets)}組\n")

    print("【制約チェック】")
    r = gen.sets
    overlap_cap = 3 if ev_mode else MODEL_PARAMS[loto]['max_overlap']
    print(f"  {'v' if all(len(n) == config['pick'] for _, n in r) else 'x'} 範囲・個数")
    print(f"  {'v' if all(len(set(a) & set(b)) <= overlap_cap for (_, a), (_, b) in combinations(r, 2)) else 'x'} 組間共通{overlap_cap}個以下")
    print(f"  {'v' if all(not has_triple_consecutive(n) for _, n in r) else 'x'} 3連番なし")
    if not ev_mode:
        ob = config["odd_even_base"][0]
        print(f"  {'v' if all(abs(sum(1 for n in ns if n % 2 == 1) - ob) <= 1 for _, ns in r) else 'x'} 奇偶バランス")
        tc = sum(sum(1 for n in ns if stats[n].category == 'COLD') for _, ns in r)
        cm, cx = _cold_range(num_sets, loto, params_map=MODEL_PARAMS)
        print(f"  {'v' if cm <= tc <= cx else '~'} COLD合計: {tc}個（目安{cm}-{cx}）")
    if not gen.fully_satisfied:
        print("  ※ 全体制約を完全には満たせませんでした（ベスト結果）")
    print()

    print("【数字のみ（コピー用）】")
    for _, nums in gen.sets:
        print(" ".join(str(n) for n in nums))

# --- v5.3 命中率特化拡張 ----------------------------------------------------
# 重要: 期待ヒット数そのものはどの5口でもほぼ同じです。
# この拡張が改善するのは「5口のうち少なくとも1口が3個以上に届く確率」で、
# そのために組間重複を強く抑えた coverage-first ポートフォリオを生成します。

HITPROB_SAMPLES = {
    "loto6": 30000,
    "loto7": 40000,
}


def _hitprob_band_limits(cfg):
    lo, hi = cfg["range"]
    thirds = (hi - lo + 1) // 3
    low_max = lo + thirds - 1
    mid_max = lo + 2 * thirds - 1
    return low_max, mid_max


def _enumerate_shape_valid_candidates(cfg, num_samples, seed=0):
    """Deterministically sample shape-balanced sets with NO history dependency.

    Filters: odd_count within ±1 of target, each of 3 bands present, no 3-consecutive,
    band imbalance ≤2. Uses a fixed seed — this is a coverage-first tool, not a
    forecasting one, so the output is identical run-to-run.
    """
    lo, hi = cfg["range"]
    pick = cfg["pick"]
    pool = list(range(lo, hi + 1))
    target_odd = cfg["odd_even_base"][0]
    low_max, mid_max = _hitprob_band_limits(cfg)
    rng = random.Random(seed)

    seen = set()
    candidates = []
    max_attempts = num_samples * 30
    attempts = 0
    while len(candidates) < num_samples and attempts < max_attempts:
        attempts += 1
        nums = tuple(sorted(rng.sample(pool, pick)))
        if nums in seen:
            continue
        seen.add(nums)
        if has_triple_consecutive(nums):
            continue
        odd = sum(1 for n in nums if n % 2 == 1)
        if abs(odd - target_odd) > 1:
            continue
        b_low = sum(1 for n in nums if n <= low_max)
        b_mid = sum(1 for n in nums if low_max < n <= mid_max)
        b_high = sum(1 for n in nums if n > mid_max)
        if min(b_low, b_mid, b_high) == 0:
            continue
        if max(b_low, b_mid, b_high) - min(b_low, b_mid, b_high) > 2:
            continue
        candidates.append({"nums": nums, "odd": odd, "bands": (b_low, b_mid, b_high)})

    # Stable sort by shape compactness: closer to target_odd first, then balanced bands.
    candidates.sort(key=lambda c: (
        abs(c["odd"] - target_odd),
        max(c["bands"]) - min(c["bands"]),
        sum(c["nums"]),
    ))
    return candidates


def _find_disjoint_portfolio(candidates, num_sets, node_budget=2_000_000):
    """Backtrack to find num_sets mutually-disjoint sets.

    Returns (portfolio, nodes_visited). portfolio is None if not found
    within node_budget.
    """
    nodes = [0]
    result = [None]

    def backtrack(chosen, used, start):
        if nodes[0] >= node_budget:
            return False
        nodes[0] += 1
        if len(chosen) == num_sets:
            result[0] = list(chosen)
            return True
        for i in range(start, len(candidates)):
            cand = candidates[i]
            nset = set(cand["nums"])
            if nset & used:
                continue
            chosen.append(cand)
            if backtrack(chosen, used | nset, i + 1):
                return True
            chosen.pop()
        return False

    backtrack([], set(), 0)
    return result[0], nodes[0]


_HITPROB_DEFAULT_LABELS = ("王道A", "王道B", "分散A", "分散B", "逆張り")


def _assign_hitprob_labels(portfolio, num_sets):
    """Assign shape-sorted labels without history dependency.

    Sort by (ascending sum, then ascending odd) so ordering is deterministic
    and interpretable.
    """
    labels = list(_HITPROB_DEFAULT_LABELS[:num_sets])
    indexed = sorted(range(len(portfolio)), key=lambda i: (sum(portfolio[i]["nums"]), portfolio[i]["odd"]))
    out = [None] * len(portfolio)
    for rank, idx in enumerate(indexed):
        out[idx] = (labels[rank] if rank < len(labels) else f"組{rank+1}", portfolio[idx]["nums"])
    return out


def generate_hitprob_from_draws(draws, loto, num_sets=5, params_map=None, portfolio_map=None, seed=0):
    """Coverage-first generator: build a mutually-disjoint shape-balanced portfolio.

    History-independent. The result depends only on (loto, num_sets, seed).
    `draws` is accepted for signature compatibility with other generators but
    is not used.

    Honest caveat: expected total hits is invariant (independent draws). What
    this changes is the probability that "at least one of the 5 sets has >=3
    hits" — by reducing inter-set overlap, the portfolio's union is enlarged.
    For loto6 (5 x 6 = 30 ≤ 43) and loto7 (5 x 7 = 35 ≤ 37), a fully disjoint
    portfolio is always feasible; this builder returns one.
    """
    del draws, params_map, portfolio_map  # unused — kept for call-site compat
    cfg = LOTO_CONFIG[loto]
    num_samples = HITPROB_SAMPLES.get(loto, 20000)
    candidates = _enumerate_shape_valid_candidates(cfg, num_samples=num_samples, seed=seed)
    if not candidates:
        raise RuntimeError("命中率特化候補生成失敗")
    portfolio, nodes = _find_disjoint_portfolio(candidates, num_sets)
    if portfolio is None:
        raise RuntimeError(
            f"完全非重複ポートフォリオ構築失敗（candidates={len(candidates)}, nodes={nodes}）"
        )
    labeled = _assign_hitprob_labels(portfolio, num_sets)
    gen = GenerateResult(labeled, True)
    return None, gen, portfolio


def exact_hitprob(portfolio, loto):
    """Exact portfolio-level hit probabilities via full enumeration.

    loto6: C(43,6) = 6,096,454 combinations.
    loto7: C(37,7) = 10,295,472 combinations.

    No randomness; results are deterministic and reproducible bit-for-bit.
    """
    cfg = LOTO_CONFIG[loto]
    lo, hi = cfg["range"]
    pick = cfg["pick"]
    pool = list(range(lo, hi + 1))
    port_sets = [frozenset(nums) for nums in portfolio]

    total = 0
    any3 = any4 = any5 = 0
    total_hits_sum = 0
    for win in combinations(pool, pick):
        win_set = set(win)
        total += 1
        max_hit = 0
        per_total = 0
        for nums in port_sets:
            h = len(nums & win_set)
            per_total += h
            if h > max_hit:
                max_hit = h
        total_hits_sum += per_total
        if max_hit >= 3:
            any3 += 1
        if max_hit >= 4:
            any4 += 1
        if max_hit >= 5:
            any5 += 1

    overlaps = [len(a & b) for i, a in enumerate(port_sets) for b in port_sets[i + 1:]]
    return {
        "mean_total_hits": total_hits_sum / total,
        "any3": any3 / total,
        "any4": any4 / total,
        "any5": any5 / total,
        "union_size": len(set().union(*portfolio)),
        "avg_pair_overlap": _mean(overlaps),
        "max_pair_overlap": max(overlaps) if overlaps else 0,
        "total_combinations": total,
        "method": "exact",
    }


def estimate_hitprob(portfolio, loto, trials=100000, seed=0):
    """Monte Carlo estimator. Kept for back-compat; prefer `exact_hitprob`.

    Returns the same keys as `exact_hitprob` (with `method="mc"`) so callers
    can be swapped without conditional logic.
    """
    cfg = LOTO_CONFIG[loto]
    lo, hi = cfg["range"]
    pick = cfg["pick"]
    pool = list(range(lo, hi + 1))
    port_sets = [set(nums) for nums in portfolio]
    rng = random.Random(seed)

    any3 = any4 = any5 = 0
    total_hits = 0
    for _ in range(trials):
        win = set(rng.sample(pool, pick))
        max_hit = 0
        per_total = 0
        for nums in port_sets:
            h = len(nums & win)
            per_total += h
            if h > max_hit:
                max_hit = h
        total_hits += per_total
        if max_hit >= 3:
            any3 += 1
        if max_hit >= 4:
            any4 += 1
        if max_hit >= 5:
            any5 += 1

    overlaps = [len(set(a) & set(b)) for i, a in enumerate(portfolio) for b in portfolio[i + 1:]]
    return {
        "mean_total_hits": total_hits / trials,
        "any3": any3 / trials,
        "any4": any4 / trials,
        "any5": any5 / trials,
        "union_size": len(set().union(*portfolio)),
        "avg_pair_overlap": _mean(overlaps),
        "max_pair_overlap": max(overlaps) if overlaps else 0,
        "total_combinations": trials,
        "method": "mc",
    }


def compare_coverage_vs_hitprob(draws, loto, num_sets=5, method="exact"):
    """Apples-to-apples comparison of legacy coverage vs hitprob mode.

    method: "exact" (full enumeration, deterministic) or "mc" (Monte Carlo).
    """
    _, cov_gen, _ = generate_from_draws(
        draws, loto, num_sets=num_sets,
        params_map=MODEL_PARAMS, selection_mode="coverage",
        portfolio_map=PORTFOLIO_PARAMS, ev_mode=False,
    )
    _, hit_gen, _ = generate_hitprob_from_draws(draws, loto, num_sets=num_sets)
    cov_port = [nums for _, nums in cov_gen.sets]
    hit_port = [nums for _, nums in hit_gen.sets]
    probe = exact_hitprob if method == "exact" else estimate_hitprob
    return {
        "coverage": {"portfolio": cov_port, "estimate": probe(cov_port, loto)},
        "hitprob": {"portfolio": hit_port, "estimate": probe(hit_port, loto)},
    }


def run_hitprob(draws, loto, num_sets=5, method="exact"):
    """Run hitprob mode and print a coverage-first report with exact probabilities."""
    if len(draws) < 10:
        print(f"エラー: データ不足（{len(draws)}回）")
        return

    _, gen, _ = generate_hitprob_from_draws(draws, loto, num_sets=num_sets)
    portfolio = [nums for _, nums in gen.sets]
    probe = exact_hitprob if method == "exact" else estimate_hitprob
    est = probe(portfolio, loto)

    label_method = "exact" if est["method"] == "exact" else "Monte Carlo"
    print(f"期間: 第{draws[-1].number}回〜第{draws[0].number}回（{len(draws)}回分）")
    print("【戦略】命中率特化（coverage-first / 履歴非依存、完全非重複5口）")
    print(f"  ユニーク数: {est['union_size']}  平均組間重複: {est['avg_pair_overlap']:.2f}  最大組間重複: {est['max_pair_overlap']}")
    print(f"  {label_method}確率: 3個以上1本={100*est['any3']:.2f}%  4個以上1本={100*est['any4']:.2f}%  5個以上1本={100*est['any5']:.3f}%")
    print(f"  期待ヒット数合計: {est['mean_total_hits']:.3f}（戦略非依存、理論値と一致）")
    print()

    print(f"{'組':>2} | {'タイプ':>6} | {'数字':<34} | {'合計':>4} | {'奇:偶':>5}")
    print("-" * 64)
    for i, (label, nums) in enumerate(gen.sets, 1):
        s = sum(nums)
        odd = sum(1 for n in nums if n % 2 == 1)
        print(f"{i:2d} | {label:>6} | {' '.join(f'{n:2d}' for n in nums):<34} | {s:4d} | {odd}:{len(nums)-odd:<3}")
    print()

    print("【数字のみ（コピー用）】")
    for nums in portfolio:
        print(" ".join(str(n) for n in nums))


if __name__ == "__main__":
    loto = _sys.argv[1] if len(_sys.argv) > 1 else "loto6"
    csv_path = _sys.argv[2] if len(_sys.argv) > 2 else f"{loto}_data.csv"
    mode = _sys.argv[3] if len(_sys.argv) > 3 else "hitprob"
    with open(csv_path) as f:
        draws = parse_csv(f.read(), loto)
    if mode == "hitprob":
        run_hitprob(draws, loto)
    elif mode == "coverage":
        run(draws, loto, num_sets=5, ev_mode=False)
    elif mode == "ev":
        run(draws, loto, num_sets=5, ev_mode=True)
    elif mode == "compare":
        result = compare_coverage_vs_hitprob(draws, loto)
        for k in ("coverage", "hitprob"):
            est = result[k]["estimate"]
            print(f"[{k}] union={est['union_size']} avg_overlap={est['avg_pair_overlap']:.2f} any3={100*est['any3']:.2f}% any4={100*est['any4']:.2f}%")
            for nums in result[k]["portfolio"]:
                print("  " + " ".join(str(n) for n in nums))
            print()
    else:
        raise SystemExit("mode must be one of: hitprob / coverage / ev / compare")
