"""Offline optimizer for the extended hitprob portfolios (overlap structures).

`loto_predictor_chatgpt._PRECOMPUTED_EXTENDED_PORTFOLIOS` のテーブルは
このスクリプトで導出した。再現・再検証・上限変更時の再生成に使う。

理論的背景:
  any3 確率（少なくとも1口が3個以上ヒット）は、各数字が「どのチケット集合に
  属するか」(membership mask) の多重集合だけで決まり、具体的な数字には依存
  しない。完全非重複上限を超える組数では合計スロット数 m*pick がプールサイズ
  N を超える。E = m*pick - N は全N数字を使う場合の最小重複スロット数で、
  一般には未使用数字を含む membership-mask 多重集合全体が探索対象になる。

探索結果（2026-06-11）:
  - 10口以下の best-found 構造は「相異なるペアの単純グラフを最小サポート k
    (= min{k: C(k,2) >= E, 次数 <= pick}) のチケット部分集合に密に載せ、
    補グラフが1頂点に集中（K_k マイナス スター）になる形」だった。
  - 探索記録上、loto7 6口は全構造空間（マスクサイズ2-6, 未使用数字0-2）、
    loto6 8口はマスクサイズ2-3・未使用数字0-1 を全数探索して最適性を確認。
  - 他の組数は dense-family 全数探索 + 多スタート焼きなましの一致で確認
   （独立シードがすべて同一値に収束）。

使い方:
  python3 optimize_hitprob_extended.py family <loto> <num_sets> <k>
  python3 optimize_hitprob_extended.py anneal <loto> <num_sets> <seed> <minutes>
  python3 optimize_hitprob_extended.py shrink <loto> <num_sets>
      （num_sets+1口の格納済み best-found から各1口を削除し、全候補を exact
        評価する。v5.11 で loto7 13/17口・loto6 19口を改善）
  python3 optimize_hitprob_extended.py emit
      （既知の最良構造から確定テーブルを再構築し、本番DPで fail3 を検証して
        Python リテラルを出力する）
  python3 optimize_hitprob_extended.py emit-plus
      （11-20口の best-found 構造 BEST_PLUS_MASKS から
        _PRECOMPUTED_BEST_FOUND_PORTFOLIOS を再構築・検証・出力する。
        v5.9-v5.12。全空間の最適性証明なし）
  python3 optimize_hitprob_extended.py retune <loto> <num_sets> [budget_s] [starts]
      （v5.12。共通乱数 MC で広く探索し差分 exact polish で仕上げる。
        採用条件は fail3 の厳密改善かつ loto7 は exact 入賞確率の非悪化。
        best-found は最適性証明がないので、回すたびに改善しうる）
  python3 optimize_hitprob_extended.py prize-prob <loto> <num_sets>
      （格納済み構造の exact 入賞確率。loto7 は本数字>=3 かつ 本数字+ボーナス>=4、
        loto6 は入賞 <=> any3）

⚠ retune / prize-prob / polish_swap_exact / exact_prize_prob は **numpy>=2.0 が必要**
  （`np.bitwise_count` は numpy 2.0 で追加された。関数内 import なので、
    本番の loto_predictor_chatgpt は numpy 非依存のまま）。
  無い環境では `uv run --with 'numpy>=2.0' python3 optimize_hitprob_extended.py ...`。
"""
import json
import math
import random
import sys
import time
from itertools import combinations
from math import comb

import loto_predictor_chatgpt as lp


# ---------------------------------------------------------------------------
# Exact evaluation on mask structures (fast: finished tickets marginalized)
# ---------------------------------------------------------------------------

def portfolio_to_masks(portfolio):
    """Concrete portfolio (list of number-lists) -> {membership mask: count}."""
    masks = {}
    allnums = set().union(*(set(s) for s in portfolio))
    for n in sorted(allnums):
        mk = 0
        for i, s in enumerate(portfolio):
            if n in s:
                mk |= 1 << i
        masks[mk] = masks.get(mk, 0) + 1
    return masks


def fail_count(masks, m, n_pool, pick, threshold=3):
    """Exact count of draws where every ticket has < threshold hits.

    Equivalent to lp._fail_count_under_threshold but operates on a mask
    structure and marginalizes the hit dimension of finished tickets during
    the DP, which keeps the state space small for dense 9-10 ticket cases.
    """
    counts = [(mk, c) for mk, c in masks.items() if c > 0]
    z = n_pool - sum(c for _, c in counts)
    assert z >= 0, "structure uses more numbers than the pool"

    # order masks greedily to keep the active-ticket set small
    remaining = list(range(len(counts)))
    order = []
    active = set()
    while remaining:
        best_i, best_key = None, None
        for i in remaining:
            bits = {b for b in range(m) if counts[i][0] & (1 << b)}
            other = 0
            for j in remaining:
                if j != i:
                    other |= counts[j][0]
            still = {b for b in (active | bits) if other & (1 << b)}
            key = (len(still), len(active | bits))
            if best_key is None or key < best_key:
                best_key, best_i = key, i
        order.append(best_i)
        active |= {b for b in range(m) if counts[best_i][0] & (1 << b)}
        remaining.remove(best_i)
        other = 0
        for j in remaining:
            other |= counts[j][0]
        active = {b for b in active if other & (1 << b)}

    seq = [counts[i] for i in order]
    last_idx = {}
    for idx, (mk, _c) in enumerate(seq):
        for b in range(m):
            if mk & (1 << b):
                last_idx[b] = idx

    active_list = []
    dp = {(0, ()): 1}
    for idx, (mk, c) in enumerate(seq):
        bits = [b for b in range(m) if mk & (1 << b)]
        for b in bits:
            if b not in active_list:
                active_list.append(b)
                dp = {(sel, hits + (0,)): w for (sel, hits), w in dp.items()}
        pos = [active_list.index(b) for b in bits]
        choices = [comb(c, t) for t in range(min(c, pick) + 1)]
        new = {}
        for (sel, hits), ways in dp.items():
            room = min(threshold - 1 - hits[p] for p in pos)
            for take in range(min(c, pick - sel, room) + 1):
                if take == 0:
                    key = (sel, hits)
                    new[key] = new.get(key, 0) + ways
                    continue
                h = list(hits)
                for p in pos:
                    h[p] += take
                key = (sel + take, tuple(h))
                new[key] = new.get(key, 0) + ways * choices[take]
        dp = new
        finished = [b for b in active_list if last_idx[b] == idx]
        if finished:
            drop = sorted((active_list.index(b) for b in finished), reverse=True)
            merged = {}
            for (sel, hits), ways in dp.items():
                h = list(hits)
                for p in drop:
                    del h[p]
                key = (sel, tuple(h))
                merged[key] = merged.get(key, 0) + ways
            dp = merged
            for b in finished:
                active_list.remove(b)

    return sum(w * comb(z, pick - sel) for (sel, _h), w in dp.items())


# ---------------------------------------------------------------------------
# Structure <-> concrete portfolio
# ---------------------------------------------------------------------------

def edges_to_masks(edges, m, pick):
    """Simple-graph excess structure -> full mask dict (with singletons)."""
    deg = [0] * m
    masks = {}
    for a, b in edges:
        mk = (1 << a) | (1 << b)
        masks[mk] = masks.get(mk, 0) + 1
        deg[a] += 1
        deg[b] += 1
    for i in range(m):
        x = pick - deg[i]
        assert x >= 0, "degree exceeds pick"
        if x:
            masks[1 << i] = masks.get(1 << i, 0) + x
    return masks


def masks_to_portfolio(masks, m, n_pool, lo=1):
    """Deterministic concrete portfolio from a mask structure.

    The exact probability is invariant under which concrete numbers fill which
    mask, so this only spreads numbers across the range for readability.
    """
    items = []
    for mk in sorted(masks, key=lambda mk: (-bin(mk).count("1"), mk)):
        items.extend([mk] * masks[mk])
    pool = list(range(lo, lo + n_pool))
    used = len(items)
    seen = set()
    positions = []
    for i in range(used):
        p = round(i * n_pool / used)
        while p in seen:
            p += 1
        if p >= n_pool:
            p = next(q for q in range(n_pool) if q not in seen)
        seen.add(p)
        positions.append(p)
    tickets = [[] for _ in range(m)]
    for mk, p in zip(items, positions):
        for i in range(m):
            if mk & (1 << i):
                tickets[i].append(pool[p])
    return [tuple(sorted(t)) for t in tickets]


def shape_polish(port, cfg):
    """Cosmetic balancing: swap concrete numbers between membership masks.

    Any permutation of numbers across masks preserves the overlap structure
    and hence the exact probabilities; this descent only improves the
    aesthetic shape objective (odd/even, sum balance, bands, no runs of 3).
    """
    port = [list(s) for s in port]
    m = len(port)
    nums = sorted(set().union(*(set(s) for s in port)))
    mask_of = {}
    for n in nums:
        mask_of[n] = sum(1 << i for i, s in enumerate(port) if n in s)

    def apply_swap(a, b):
        for i in range(m):
            sa, sb = a in port[i], b in port[i]
            if sa == sb:
                continue
            if sa:
                port[i].remove(a)
                port[i].append(b)
            else:
                port[i].remove(b)
                port[i].append(a)

    best = lp._shape_objective([tuple(sorted(s)) for s in port], cfg)
    improved = True
    while improved:
        improved = False
        for ai in range(len(nums)):
            for bi in range(ai + 1, len(nums)):
                a, b = nums[ai], nums[bi]
                if mask_of[a] == mask_of[b]:
                    continue
                apply_swap(a, b)
                score = lp._shape_objective([tuple(sorted(s)) for s in port], cfg)
                if score + 1e-9 < best:
                    best = score
                    mask_of[a], mask_of[b] = mask_of[b], mask_of[a]
                    improved = True
                else:
                    apply_swap(a, b)
    return [tuple(sorted(s)) for s in port]


# ---------------------------------------------------------------------------
# Searches
# ---------------------------------------------------------------------------

def family_search(loto, num_sets, k, verbose=True):
    """Exhaust the dense-simple-graph family: E excess edges on k tickets."""
    cfg = lp.LOTO_CONFIG[loto]
    lo, hi = cfg["range"]
    n_pool = hi - lo + 1
    pick = cfg["pick"]
    m = num_sets
    excess = m * pick - n_pool
    edges = list(combinations(range(k), 2))
    assert len(edges) >= excess, "k too small for a simple graph"
    best, best_edges = None, None
    for combo in combinations(edges, excess):
        deg = [0] * k
        ok = True
        for a, b in combo:
            deg[a] += 1
            deg[b] += 1
            if deg[a] > pick or deg[b] > pick:
                ok = False
                break
        if not ok or min(deg) == 0:
            continue
        f = fail_count(edges_to_masks(combo, m, pick), m, n_pool, pick)
        if best is None or f < best:
            best, best_edges = f, combo
    if verbose:
        print(f"{loto} {num_sets}口 k={k}: best fail3={best} edges={best_edges}")
    return best, best_edges


def anneal(loto, num_sets, seed, minutes, start_port=None, verbose=True):
    """Simulated annealing over concrete portfolios + final 1-swap descent."""
    cfg = lp.LOTO_CONFIG[loto]
    lo, hi = cfg["range"]
    pick = cfg["pick"]
    n_pool = hi - lo + 1
    m = num_sets
    pool = list(range(lo, hi + 1))
    rng = random.Random(seed * 7919 + 13)
    cache = {}

    def evaluate(port):
        key = frozenset(portfolio_to_masks(port).items())
        v = cache.get(key)
        if v is None:
            v = fail_count(dict(key), m, n_pool, pick)
            cache[key] = v
        return v

    if start_port is None:
        base = [list(d["nums"]) for d in
                lp._balanced_disjoint_portfolio(loto, num_sets=lp._max_disjoint_sets(loto))]
        leftovers = sorted(set(pool) - set().union(*(set(s) for s in base)))
        n_extra = m - len(base)
        extras = [[] for _ in range(n_extra)]
        for idx, n in enumerate(leftovers):
            extras[idx % n_extra].append(n)
        for nums in extras:
            cands = [n for n in pool if n not in nums]
            rng.shuffle(cands)
            nums.extend(cands[: pick - len(nums)])
        start_port = base + extras
    port = [list(s) for s in start_port]

    t_total = minutes * 60.0
    t_end = time.time() + t_total
    anneal_end = time.time() + t_total * 0.7
    cur = evaluate(port)
    best, best_port = cur, [list(s) for s in port]

    deltas = []
    for _ in range(30):
        i, j = rng.randrange(m), rng.randrange(pick)
        v = rng.choice([x for x in pool if x not in port[i]])
        old = port[i][j]
        port[i][j] = v
        d = abs(evaluate(port) - cur)
        port[i][j] = old
        if d:
            deltas.append(d)
    t0_temp = (sorted(deltas)[len(deltas) // 2] if deltas else 1000) * 2.0

    while time.time() < anneal_end:
        frac = max(0.0, (anneal_end - time.time()) / (t_total * 0.7))
        t_temp = t0_temp * (0.001 ** (1 - frac))
        i, j = rng.randrange(m), rng.randrange(pick)
        v = rng.choice([x for x in pool if x not in port[i]])
        old = port[i][j]
        port[i][j] = v
        cand = evaluate(port)
        delta = cand - cur
        if delta <= 0 or rng.random() < math.exp(-delta / max(t_temp, 1e-9)):
            cur = cand
            if cur < best:
                best, best_port = cur, [sorted(s) for s in port]
        else:
            port[i][j] = old

    # Final 1-swap descent. The deadline is checked per evaluation: a full
    # sweep is m*pick*|pool| evaluations, so checking only per sweep (as before
    # v5.9) overran the budget by ~30 min at 15 tickets and by hours at 20.
    port = [list(s) for s in best_port]
    improved = True
    timed_out = False
    while improved and not timed_out:
        improved = False
        for i in range(m):
            for j in range(pick):
                old = port[i][j]
                others = set(port[i]) - {old}
                for v in pool:
                    if time.time() >= t_end:
                        timed_out = True
                        break
                    if v == old or v in others:
                        continue
                    port[i][j] = v
                    cand = evaluate(port)
                    if cand < best:
                        best, old, improved = cand, v, True
                    else:
                        port[i][j] = old
                if timed_out:
                    break
            if timed_out:
                break
    if verbose:
        print(f"{loto} {num_sets}口 seed={seed}: fail3={best}")
    return best, [tuple(sorted(s)) for s in port]


# ---------------------------------------------------------------------------
# Known-best excess structures and table emission
# ---------------------------------------------------------------------------

# Excess edges (ticket-index pairs) of the best structures found per case.
# K4-minus-edge for E=5; K_k minus a star for the denser cases. See module
# docstring for how each was verified.
BEST_EXCESS_EDGES = {
    ("loto6", 8): [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)],
    ("loto6", 9): [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3),
                   (1, 4), (2, 3), (2, 4), (3, 4)],
    ("loto6", 10): [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 2),
                    (1, 3), (1, 4), (1, 5), (1, 6), (2, 3), (2, 4), (2, 5),
                    (3, 4), (3, 5), (4, 5)],
    ("loto7", 6): [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)],
    ("loto7", 7): [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3),
                   (1, 4), (1, 5), (2, 3), (2, 4), (3, 4)],
    ("loto7", 8): [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 2),
                   (1, 3), (1, 4), (1, 5), (1, 6), (2, 3), (2, 4), (2, 5),
                   (2, 6), (3, 4), (3, 5), (3, 6), (4, 5)],
    ("loto7", 9): [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
                   (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 3),
                   (2, 4), (2, 5), (2, 6), (2, 7), (3, 4), (3, 5), (3, 6),
                   (3, 7), (4, 5), (4, 6), (4, 7), (5, 6)],
    # Densest case (E=33 on 10 tickets, support must be all 10). The best found is
    # not the balanced "minus a star" graph but a near-K9 graph plus one sparse
    # ticket: degree sequence [3, 7, 7, 7, 7, 7, 7, 7, 7, 7]. Found by a
    # complement-space hill climb (200 random starts) and confirmed against
    # multi-start annealing on concrete portfolios.
    ("loto7", 10): [(0, 1), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 9),
                    (1, 5), (1, 7), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
                    (2, 8), (2, 9), (3, 4), (3, 5), (3, 6), (3, 8), (3, 9),
                    (4, 5), (4, 6), (4, 8), (4, 9), (5, 7), (5, 8), (6, 7),
                    (6, 8), (6, 9), (7, 8), (7, 9), (8, 9)],
}


# Best-found mask structures for 11-20 sets (拡張プラスモード, v5.9-v5.11).
# Some cases require numbers shared by 3+ tickets; even where pair-only overlap
# is feasible, the full structure space is too large to exhaust. Structures are
# stored as {membership mask: count} over ticket indices.
# Originally derived by multi-start simulated annealing (2 seeds per case, best
# kept); v5.11 replaces loto7 13/17口 and loto6 19口 via next-size shrinking.
# NO global optimality proof.
BEST_PLUS_MASKS = {
    ('loto6', 11): {1: 5, 6: 1, 8: 6, 18: 1, 20: 1, 34: 1, 36: 1, 48: 1, 66: 1, 68: 1, 80: 1, 96: 1, 128: 4, 129: 1, 258: 1, 260: 1, 272: 1, 288: 1, 320: 1, 512: 5, 640: 1, 1026: 1, 1028: 1, 1040: 1, 1056: 1, 1088: 1, 1280: 1},
    ('loto6', 12): {2: 3, 4: 1, 5: 1, 8: 2, 10: 1, 16: 3, 24: 1, 32: 2, 34: 1, 40: 1, 48: 1, 65: 1, 68: 1, 129: 1, 132: 1, 192: 1, 256: 2, 258: 1, 264: 1, 272: 1, 288: 1, 512: 1, 513: 1, 576: 1, 640: 1, 1025: 1, 1028: 1, 1088: 1, 1152: 1, 1536: 1, 2049: 1, 2052: 1, 2112: 1, 2176: 1, 2560: 1, 3072: 1},
    ('loto6', 13): {1: 1, 2: 1, 3: 1, 8: 2, 9: 1, 10: 1, 20: 1, 36: 1, 48: 1, 68: 1, 80: 1, 96: 1, 128: 2, 129: 1, 130: 1, 260: 1, 272: 1, 288: 1, 320: 1, 512: 1, 513: 1, 514: 1, 520: 1, 640: 1, 1024: 1, 1025: 1, 1026: 1, 1032: 1, 1152: 1, 1536: 1, 2052: 1, 2064: 1, 2080: 1, 2112: 1, 2304: 1, 4100: 1, 4112: 1, 4128: 1, 4160: 1, 4352: 1, 6144: 1},
    ('loto6', 14): {5: 1, 8: 1, 9: 1, 12: 1, 17: 1, 20: 1, 24: 1, 32: 1, 33: 1, 36: 1, 48: 1, 66: 1, 130: 1, 192: 1, 257: 1, 260: 1, 264: 1, 272: 1, 288: 1, 514: 1, 576: 1, 640: 1, 1026: 1, 1088: 1, 1152: 1, 1536: 1, 2049: 1, 2052: 1, 2056: 1, 2064: 1, 2080: 1, 2304: 1, 4098: 1, 4160: 1, 4224: 1, 4608: 1, 5120: 1, 8194: 1, 8256: 1, 8320: 1, 8704: 1, 9216: 1, 12288: 1},
    ('loto6', 15): {5: 1, 9: 1, 10: 1, 17: 1, 18: 1, 33: 1, 34: 1, 52: 1, 132: 1, 160: 1, 192: 1, 280: 1, 320: 1, 513: 1, 516: 1, 520: 1, 640: 1, 768: 1, 1026: 1, 1040: 1, 1280: 1, 2052: 1, 2112: 1, 2176: 1, 2560: 1, 3072: 1, 4104: 1, 4128: 1, 4160: 1, 4354: 1, 5120: 1, 8194: 1, 8200: 1, 8256: 1, 8448: 1, 9216: 1, 12304: 1, 16385: 1, 16388: 1, 16416: 1, 16448: 1, 16512: 1, 18432: 1},
    ('loto7', 11): {9: 1, 10: 1, 18: 1, 21: 1, 24: 1, 33: 1, 34: 1, 36: 1, 48: 1, 65: 1, 66: 1, 68: 1, 72: 1, 130: 1, 132: 1, 136: 1, 144: 1, 160: 1, 257: 1, 260: 1, 272: 1, 288: 1, 320: 1, 384: 1, 518: 1, 520: 1, 544: 1, 576: 1, 641: 1, 768: 1, 1025: 1, 1026: 1, 1028: 1, 1032: 1, 1040: 1, 1088: 1, 1536: 1},
    ('loto7', 12): {3: 1, 5: 1, 12: 1, 24: 1, 34: 1, 40: 1, 52: 1, 65: 1, 66: 1, 80: 1, 130: 1, 132: 1, 136: 1, 145: 1, 257: 1, 258: 1, 272: 1, 288: 1, 320: 1, 514: 1, 516: 1, 520: 1, 545: 1, 704: 1, 1028: 1, 1032: 1, 1042: 1, 1120: 1, 1152: 1, 1792: 1, 2052: 1, 2056: 1, 2112: 1, 2208: 1, 2304: 1, 2576: 1, 3073: 1},
    ('loto7', 13): {9: 1, 10: 1, 18: 1, 20: 1, 38: 1, 65: 1, 72: 1, 96: 1, 132: 1, 145: 1, 160: 1, 259: 1, 264: 1, 272: 1, 448: 1, 516: 1, 544: 1, 578: 1, 768: 1, 1028: 1, 1032: 1, 1057: 1, 1104: 1, 1664: 1, 2056: 1, 2096: 1, 2176: 1, 2308: 1, 2561: 1, 3074: 1, 4101: 1, 4104: 1, 4128: 1, 4226: 1, 4624: 1, 5376: 1, 6208: 1},
    ('loto7', 14): {10: 1, 12: 1, 40: 1, 88: 1, 98: 1, 148: 1, 161: 1, 259: 1, 260: 1, 264: 1, 448: 1, 518: 1, 520: 1, 528: 1, 577: 1, 1025: 1, 1042: 1, 1088: 1, 1312: 1, 2065: 1, 2084: 1, 2184: 1, 2816: 1, 3200: 1, 4101: 1, 4226: 1, 4368: 1, 4640: 1, 5120: 1, 6208: 1, 8193: 1, 8240: 1, 8260: 1, 8832: 1, 9216: 1, 10242: 1, 12288: 1},
    ('loto7', 15): {12: 1, 26: 1, 34: 1, 81: 1, 129: 1, 160: 1, 288: 1, 322: 1, 518: 1, 545: 1, 704: 1, 776: 1, 1029: 1, 1120: 1, 1296: 1, 2096: 1, 2120: 1, 2308: 1, 3200: 1, 4105: 1, 4164: 1, 4240: 1, 5632: 1, 6146: 1, 8200: 1, 8324: 1, 8720: 1, 9218: 1, 10241: 1, 12544: 1, 16404: 1, 16514: 1, 16641: 1, 17416: 1, 18944: 1, 20512: 1, 24640: 1},
    ('loto6', 16): {6: 1, 12: 1, 48: 1, 66: 1, 81: 1, 129: 1, 136: 1, 160: 1, 258: 1, 260: 1, 521: 1, 528: 1, 608: 1, 640: 1, 1026: 1, 1028: 1, 1088: 1, 1280: 1, 2064: 1, 2088: 1, 2240: 1, 2560: 1, 4120: 1, 4128: 1, 4352: 1, 6145: 1, 8225: 1, 8264: 1, 8336: 1, 8704: 1, 10240: 1, 12288: 1, 16385: 1, 16386: 1, 16388: 1, 16640: 1, 17408: 1, 32770: 1, 32772: 1, 33024: 1, 33792: 1, 36864: 1, 49152: 1},
    ('loto6', 17): {3: 1, 9: 1, 65: 1, 72: 1, 129: 1, 164: 1, 192: 1, 257: 1, 258: 1, 260: 1, 264: 1, 320: 1, 384: 1, 520: 1, 532: 1, 1040: 1, 1058: 1, 1152: 1, 2050: 1, 4226: 1, 4640: 1, 5120: 1, 6160: 1, 8194: 1, 8240: 1, 8704: 1, 10244: 1, 16385: 1, 16392: 1, 16400: 1, 16448: 1, 18432: 1, 32776: 1, 33796: 1, 34848: 1, 45056: 1, 49664: 1, 65568: 1, 65600: 1, 68096: 1, 69636: 1, 74752: 1, 98320: 1},
    ('loto6', 18): {36: 1, 49: 1, 70: 1, 138: 1, 148: 1, 224: 1, 274: 1, 288: 1, 776: 1, 1025: 1, 1536: 1, 2049: 1, 2064: 1, 3072: 1, 4672: 1, 5120: 1, 6144: 1, 8204: 1, 8224: 1, 8512: 1, 8832: 1, 16464: 1, 16768: 1, 16900: 1, 20488: 1, 24578: 1, 32840: 1, 32897: 1, 33028: 1, 33282: 1, 40976: 1, 49184: 1, 65537: 1, 65544: 1, 66560: 1, 67584: 1, 69632: 1, 131073: 1, 131074: 1, 132096: 1, 133120: 1, 135168: 1, 196608: 1},
    ('loto6', 19): {3: 1, 33: 1, 34: 1, 100: 1, 140: 1, 280: 1, 552: 1, 704: 1, 1153: 1, 1792: 1, 2128: 1, 2208: 1, 2564: 1, 4098: 1, 5184: 1, 8194: 1, 8197: 1, 8224: 1, 10248: 1, 16386: 1, 16640: 1, 20496: 1, 24576: 1, 33808: 1, 37120: 1, 49160: 1, 65537: 1, 65538: 1, 65792: 1, 73728: 1, 81920: 1, 98304: 1, 131137: 1, 131600: 1, 132100: 1, 135296: 1, 165888: 1, 262216: 1, 262288: 1, 265216: 1, 266752: 1, 294916: 1, 393472: 1},
    ('loto6', 20): {82: 1, 134: 1, 160: 1, 257: 1, 296: 1, 384: 1, 1536: 1, 2056: 1, 2177: 1, 2304: 1, 4144: 1, 4352: 1, 5128: 1, 6144: 1, 8328: 1, 10242: 1, 16912: 1, 17440: 1, 20544: 1, 24580: 1, 33284: 1, 33808: 1, 40961: 1, 65556: 1, 66050: 1, 66624: 1, 114688: 1, 131081: 1, 131168: 1, 131328: 1, 133124: 1, 135296: 1, 262149: 1, 262720: 1, 294914: 1, 335872: 1, 393232: 1, 524291: 1, 532992: 1, 540680: 1, 557120: 1, 589856: 1, 787456: 1},
    ('loto7', 16): {52: 1, 67: 1, 152: 1, 324: 1, 385: 1, 518: 1, 552: 1, 784: 1, 1029: 1, 1282: 1, 1600: 1, 2120: 1, 2180: 1, 2336: 1, 2561: 1, 4106: 1, 4164: 1, 4256: 1, 5136: 1, 8226: 1, 8384: 1, 8456: 1, 11264: 1, 12289: 1, 16393: 1, 17024: 1, 17440: 1, 18450: 1, 20736: 1, 24592: 1, 32785: 1, 32864: 1, 32898: 1, 33800: 1, 38912: 1, 41472: 1, 49156: 1},
    ('loto7', 17): {38: 1, 69: 1, 74: 1, 273: 1, 448: 1, 776: 1, 1104: 1, 1164: 1, 1569: 1, 2072: 1, 2336: 1, 4144: 1, 4226: 1, 4356: 1, 7680: 1, 8232: 1, 8450: 1, 8836: 1, 10241: 1, 16912: 1, 17410: 1, 18560: 1, 20481: 1, 24640: 1, 32786: 1, 32897: 1, 33376: 1, 34820: 1, 46080: 1, 49160: 1, 65696: 1, 66051: 1, 67648: 1, 69640: 1, 73744: 1, 81924: 1, 99584: 1},
    ('loto7', 18): {102: 1, 131: 1, 281: 1, 561: 1, 1036: 1, 1248: 1, 2436: 1, 4240: 1, 4928: 1, 6154: 1, 8272: 1, 8450: 1, 9217: 1, 10756: 1, 12320: 1, 16456: 1, 17664: 1, 18946: 1, 20481: 1, 32773: 1, 33056: 1, 35856: 1, 40968: 1, 49792: 1, 65554: 1, 65672: 1, 67072: 1, 69636: 1, 84000: 1, 98368: 1, 131624: 1, 132098: 1, 133185: 1, 139392: 1, 147476: 1, 167936: 1, 196864: 1},
    ('loto7', 19): {102: 1, 673: 1, 1072: 1, 1284: 1, 2059: 1, 2068: 1, 4242: 1, 6720: 1, 8464: 1, 9728: 1, 12293: 1, 16516: 1, 16705: 1, 17410: 1, 32968: 1, 33296: 1, 37120: 1, 40992: 1, 51200: 1, 66306: 1, 66569: 1, 67968: 1, 73800: 1, 86048: 1, 131153: 1, 131368: 1, 136320: 1, 141314: 1, 147968: 1, 229380: 1, 262668: 1, 263232: 1, 264224: 1, 282632: 1, 294915: 1, 327696: 1, 401536: 1},
    ('loto7', 20): {116: 1, 204: 1, 1409: 1, 2569: 1, 5696: 1, 6402: 1, 8289: 1, 8472: 1, 16546: 1, 20490: 1, 28676: 1, 33328: 1, 33798: 1, 43136: 1, 51201: 1, 66308: 1, 67650: 1, 69665: 1, 82960: 1, 131112: 1, 140288: 1, 147968: 1, 164096: 1, 196736: 1, 262163: 1, 262784: 1, 263176: 1, 299072: 1, 336128: 1, 393232: 1, 527392: 1, 528528: 1, 532994: 1, 540992: 1, 622600: 1, 655365: 1, 788484: 1},
}


def shrink_next_portfolio(loto, num_sets, verbose=True):
    """Find the best exact `num_sets` subset of the stored next-size portfolio."""
    source_key = (loto, num_sets + 1)
    source = lp._PRECOMPUTED_BEST_FOUND_PORTFOLIOS.get(source_key)
    if source is None:
        raise ValueError(f"格納済みの {loto} {num_sets + 1}口構造がありません")

    cfg = lp.LOTO_CONFIG[loto]
    lo, hi = cfg["range"]
    n_pool = hi - lo + 1
    pick = cfg["pick"]
    best = None
    best_port = None
    best_removed = None
    for removed in range(len(source)):
        candidate = source[:removed] + source[removed + 1:]
        fail3 = fail_count(
            portfolio_to_masks(candidate), num_sets, n_pool, pick,
        )
        if best is None or fail3 < best:
            best = fail3
            best_port = candidate
            best_removed = removed
    assert best is not None and best_port is not None and best_removed is not None
    if verbose:
        print(
            f"{loto} {num_sets}口: fail3={best} "
            f"source={num_sets + 1}口 removed_index={best_removed}"
        )
    return best, best_port, best_removed


def emit_plus_table():
    """Rebuild the best-found 11-20 set portfolios from BEST_PLUS_MASKS,
    verify against the production DP, and print the Python literal."""
    print("_PRECOMPUTED_BEST_FOUND_PORTFOLIOS = {")
    for (loto, m), masks in sorted(BEST_PLUS_MASKS.items()):
        cfg = lp.LOTO_CONFIG[loto]
        lo, hi = cfg["range"]
        pick = cfg["pick"]
        n_pool = hi - lo + 1
        port = shape_polish(masks_to_portfolio(masks, m, n_pool, lo), cfg)
        f_tool = fail_count(portfolio_to_masks(port), m, n_pool, pick)
        f_prod = lp._fail_count_under_threshold(port, loto, 3)
        assert f_tool == f_prod, (loto, m, f_tool, f_prod)
        total = comb(n_pool, pick)
        any3 = 1 - f_prod / total
        print(f"    # fail3={f_prod}, exact any3={any3*100:.4f}% (best-found)")
        print(f"    ({loto!r}, {m}): (")
        for t in port:
            print(f"        {t},")
        print("    ),")
    print("}")


# --- v5.12 retune: MC 探索 + 差分 exact polish -------------------------------
# anneal() は1評価が exact DP（loto7 20口で約1.3秒）なので試行回数を稼げない。
# 共通乱数モンテカルロ（1評価ミリ秒）で広く探索し、exact の差分評価で仕上げる。
# numpy はこの導出ツール内でのみ使う（本番の loto_predictor_chatgpt は非依存のまま）。


def _ticket_bits(ticket, lo):
    v = 0
    for n in ticket:
        v |= 1 << (n - lo)
    return v


def mc_samples(loto, n, seed):
    """本数字のみの指示行列 (n, pool)。any3 の近似評価用。"""
    import numpy as np

    cfg = lp.LOTO_CONFIG[loto]
    lo, hi = cfg["range"]
    pick = cfg["pick"]
    pool = hi - lo + 1
    rng = np.random.default_rng(seed)
    idx = rng.random((n, pool)).argsort(axis=1)[:, :pick]
    main = np.zeros((n, pool), dtype=np.int16)
    main[np.arange(n)[:, None], idx] = 1
    return main


def mc_hill_climb(portfolio, loto, budget_s, seed, samples=200_000):
    """共通乱数 MC で any3 の 1-swap 山登り。近似なので採用前に exact で採点する。

    ⚠ 打ち切りが time.time() ベースなので、**同じ seed でも実行ごとに結果が変わる**
    （マシン負荷で到達するスイープ数が変わるため）。得られた構造を残したい場合は
    retune の JSON 出力を保存すること。再現性が要るなら budget_s を十分大きくして
    収束させるか、スイープ数で打ち切る実装に変える必要がある。
    """
    import numpy as np

    cfg = lp.LOTO_CONFIG[loto]
    lo, hi = cfg["range"]
    main = mc_samples(loto, samples, seed)
    rng = np.random.default_rng(seed + 1)
    port = [list(t) for t in portfolio]
    m = len(port)

    def cols(p):
        t = np.zeros((len(p), hi - lo + 1), dtype=np.int16)
        for i, s in enumerate(p):
            for n in s:
                t[i, n - lo] = 1
        return (main @ t.T) >= 3

    wm = cols(port)
    cur = float(wm.any(axis=1).mean())
    start = time.time()
    while time.time() - start < budget_s:
        improved = False
        cnt = wm.sum(axis=1)
        for i in rng.permutation(m):
            if time.time() - start > budget_s:
                break
            others = (cnt - wm[:, i]) > 0
            in_t = set(port[i])
            best_val, best_cand = cur, None
            for x in sorted(in_t):
                for y in range(lo, hi + 1):
                    if y in in_t:
                        continue
                    cand = sorted((in_t - {x}) | {y})
                    hit = main[:, np.array(cand) - lo].sum(axis=1) >= 3
                    val = float((others | hit).mean())
                    if val > best_val + 1e-9:
                        best_val, best_cand = val, cand
            if best_cand is not None:
                port[i] = best_cand
                wm = cols(port)
                cnt = wm.sum(axis=1)
                cur = best_val
                improved = True
        if not improved:
            break
    return [tuple(sorted(t)) for t in port]


def _frontier(portfolio, loto):
    """bad_count<=1 の主数字集合と、その唯一の bad 口 index（bad0 は -1）。"""
    import numpy as np

    cfg = lp.LOTO_CONFIG[loto]
    lo, hi = cfg["range"]
    pick = cfg["pick"]
    tb = np.array([_ticket_bits(t, lo) for t in portfolio], dtype=np.int64)
    m = len(portfolio)
    keep_bits, keep_bad = [], []
    it = combinations(range(hi - lo + 1), pick)
    while True:
        block = []
        for _ in range(500_000):
            nxt = next(it, None)
            if nxt is None:
                break
            block.append(nxt)
        if not block:
            break
        arr = np.array(block, dtype=np.int64)
        mbits = np.zeros(len(arr), dtype=np.int64)
        for c in range(pick):
            mbits |= np.left_shift(np.int64(1), arr[:, c])
        bad_cnt = np.zeros(len(arr), dtype=np.int8)
        bad_idx = np.full(len(arr), -1, dtype=np.int8)
        for i in range(m):
            isbad = np.bitwise_count(np.bitwise_and(tb[i], mbits)) >= 3
            bad_cnt += isbad
            bad_idx = np.where(isbad & (bad_idx == -1), np.int8(i), bad_idx)
        sel = bad_cnt <= 1
        keep_bits.append(mbits[sel])
        keep_bad.append(np.where(bad_cnt[sel] == 0, np.int8(-1), bad_idx[sel]))
    return np.concatenate(keep_bits), np.concatenate(keep_bad)


def polish_swap_exact(portfolio, loto, max_sweeps=10, verbose=False):
    """差分 exact による 1-swap polish。

    口 i を差し替えたとき fail3 に寄与しうるのは「i を除く全口が fail」の抽選だけなので、
    その部分集合上の popcount で候補を厳密評価できる。全候補を DP に掛けるより桁違いに速い。

    返り値の converged は「改善なしのスイープで終了した」= 1-swap 局所最適に到達した、の意。
    max_sweeps に達して打ち切った場合は False で、局所最適とは断定できない。
    """
    import numpy as np

    cfg = lp.LOTO_CONFIG[loto]
    lo, hi = cfg["range"]
    port = [tuple(sorted(t)) for t in portfolio]
    m = len(port)
    cur = lp._fail_count_under_threshold(port, loto, 3)
    converged = False
    for sweep in range(max_sweeps):
        mbits, bad = _frontier(port, loto)
        improved = False
        for i in range(m):
            sub = mbits[(bad == -1) | (bad == i)]
            in_t = set(port[i])
            best_val, best_cand = cur, None
            for x in sorted(in_t):
                for y in range(lo, hi + 1):
                    if y in in_t:
                        continue
                    cand = tuple(sorted((in_t - {x}) | {y}))
                    cb = _ticket_bits(cand, lo)
                    val = int((np.bitwise_count(np.bitwise_and(cb, sub)) <= 2).sum())
                    if val < best_val:
                        best_val, best_cand = val, cand
            if best_cand is not None:
                port[i] = best_cand
                cur = best_val
                improved = True
                mbits, bad = _frontier(port, loto)
        if verbose:
            print(f"    polish sweep {sweep + 1}: fail3={cur:,}", flush=True)
        if not improved:
            converged = True
            break
    check = lp._fail_count_under_threshold(port, loto, 3)
    assert check == cur, f"差分評価と production DP が不一致: {cur} vs {check}"
    return port, cur, converged


def exact_prize_prob(portfolio, loto):
    """「N口中1口以上が入賞」の厳密確率。

    loto6: 入賞 <=> 本数字>=3 なので any3 と同値。
    loto7: 入賞 <=> a>=3 かつ a+b>=4。主数字集合 M を全列挙し、ボーナスは閉形式で処理する。
        f(M) = 0                    (max a_i >= 4)
             = C(rest-u, 2)/C(rest, 2)   (それ以外, u = |∪_{i: a_i=3}(T_i \\ M)|)
    """
    lose, denom = exact_prize_lose_count(portfolio, loto)
    return 1 - lose / denom


def exact_prize_lose_count(portfolio, loto):
    """入賞しない (主数字集合, ボーナス組) の**整数**個数と全体数を返す。

    出荷ゲートの比較は浮動小数だと丸めで順序が入れ替わりうるので、必ずこの整数で行う。
    """
    import numpy as np

    cfg = lp.LOTO_CONFIG[loto]
    lo, hi = cfg["range"]
    pick = cfg["pick"]
    nb = cfg.get("bonus_count", 0)
    pool = hi - lo + 1
    rest = pool - pick
    total = comb(pool, pick)
    denom_b = comb(rest, nb)
    if loto == "loto6":
        # ボーナスは等級判定に不要なので、負けは fail3 通りの M × 全ボーナス
        return lp._fail_count_under_threshold(portfolio, loto, 3) * denom_b, total * denom_b

    # 以降の分子 avail*(avail-1)//2 は C(avail, 2) で、ボーナス2個を前提にしている。
    # 分母 denom_b は一般形 comb(rest, nb) なので、nb が 2 以外だと黙って壊れる。
    assert nb == 2, f"loto7 経路はボーナス2個前提: bonus_count={nb}"

    tb = np.array([_ticket_bits(t, lo) for t in portfolio], dtype=np.int64)
    m = len(portfolio)
    lose = 0
    it = combinations(range(pool), pick)
    while True:
        block = []
        for _ in range(500_000):
            nxt = next(it, None)
            if nxt is None:
                break
            block.append(nxt)
        if not block:
            break
        arr = np.array(block, dtype=np.int64)
        mbits = np.zeros(len(arr), dtype=np.int64)
        for c in range(pick):
            mbits |= np.left_shift(np.int64(1), arr[:, c])
        a = np.empty((len(arr), m), dtype=np.int8)
        for i in range(m):
            a[:, i] = np.bitwise_count(np.bitwise_and(tb[i], mbits)).astype(np.int8)
        amax = a.max(axis=1)
        ubits = np.zeros(len(arr), dtype=np.int64)
        for i in range(m):
            ubits |= np.where(a[:, i] == 3, tb[i], np.int64(0))
        ubits &= ~mbits
        avail = (rest - np.bitwise_count(ubits)).astype(np.int64)
        c2 = np.where(avail >= 2, avail * (avail - 1) // 2, np.int64(0))
        lose += int(np.where(amax >= 4, np.int64(0), c2).sum())
    return lose, total * denom_b


def retune(loto, num_sets, budget_s=150.0, starts=3, seed=20260801, verbose=True):
    """出荷起点と乱数起点それぞれを MC 探索 -> exact polish に掛け、最良を返す。

    出荷起点を必ず含めるのが重要: 出荷構造が既に exact 1-swap 局所最適なケース
    （loto7 11口・loto6 11〜13口）では乱数起点が届かず、出荷起点だけが真値を示す。
    """
    import numpy as np

    cfg = lp.LOTO_CONFIG[loto]
    lo, hi = cfg["range"]
    pick = cfg["pick"]
    shipped = [tuple(sorted(t)) for t in lp._PRECOMPUTED_BEST_FOUND_PORTFOLIOS[(loto, num_sets)]]
    f_ship = lp._fail_count_under_threshold(shipped, loto, 3)
    rng = np.random.default_rng(seed)

    # 出荷起点は MC を挟まず直接 polish する経路も持つ。MC は近似なので出荷のような
    # 既に良い構造からは逆に離れてしまい、polish で戻りきらないことがある
    # （loto7 12口は「出荷を直接 polish」が最良だった）。
    cands = [("shipped+polish", shipped, False), ("shipped+mc+polish", shipped, True)]
    for k in range(starts):
        cands.append((f"random#{k + 1}",
                      [tuple(sorted(int(v) + lo for v in rng.choice(hi - lo + 1, pick, replace=False)))
                       for _ in range(num_sets)], True))
    mc_budget = budget_s / max(1, sum(1 for c in cands if c[2]))
    results = []
    for tag, start, use_mc in cands:
        seed_i = int(rng.integers(1 << 30))
        mc = mc_hill_climb(start, loto, mc_budget, seed_i) if use_mc else start
        port, f, converged = polish_swap_exact(mc, loto)
        if verbose:
            print(f"  {loto} {num_sets}口 {tag}: fail3={f:,}"
                  f"{'' if converged else ' [polish 未収束: max_sweeps 到達]'}"
                  f"{'  <<< 出荷より良い' if f < f_ship else ''}", flush=True)
        results.append((f, port, tag, converged))

    # fail3 昇順に P(win) ゲートを掛け、最初に通ったものを採る。fail3 最小を先に確定して
    # から P(win) を見ると、ゲートを通る次善候補を取りこぼす。比較は整数で行う。
    lose_ship, denom = exact_prize_lose_count(shipped, loto)
    chosen = None
    for f, port, tag, converged in sorted(results, key=lambda r: r[0]):
        if f >= f_ship:
            continue
        lose_n, _ = exact_prize_lose_count(port, loto)
        if lose_n <= lose_ship:
            chosen = (f, port, tag, lose_n, converged)
            break
        if verbose:
            print(f"  {loto} {num_sets}口 {tag}: fail3={f:,} は改善だが "
                  f"P(win) 悪化のため除外", flush=True)

    if chosen is None:
        ship_conv = next((c for f, _p, t, c in results
                          if t == "shipped+polish" and f == f_ship), False)
        if verbose:
            print(f"  => {loto} {num_sets}口 据置 fail3 {f_ship:,}  "
                  f"P(win) {100*(1-lose_ship/denom):.4f}%", flush=True)
        return {"fail3": f_ship, "shipped_fail3": f_ship,
                "win": 1 - lose_ship / denom, "shipped_win": 1 - lose_ship / denom,
                "adopt": False, "source": "shipped", "polish_converged": ship_conv,
                "portfolio": [list(t) for t in shipped]}

    f_new, port_new, src, lose_n, conv_new = chosen
    if verbose:
        print(f"  => {loto} {num_sets}口 採用 fail3 {f_ship:,}->{f_new:,}  "
              f"P(win) {100*(1-lose_ship/denom):.4f}%->{100*(1-lose_n/denom):.4f}%  [{src}]",
              flush=True)
    return {"fail3": f_new, "shipped_fail3": f_ship,
            "win": 1 - lose_n / denom, "shipped_win": 1 - lose_ship / denom,
            "adopt": True, "source": src, "polish_converged": conv_new,
            "portfolio": [list(t) for t in port_new]}


def emit_table():
    """Rebuild final portfolios from BEST_EXCESS_EDGES, verify, print literal."""
    print("_PRECOMPUTED_EXTENDED_PORTFOLIOS = {")
    for (loto, m), edges in sorted(BEST_EXCESS_EDGES.items()):
        assert edges is not None, f"missing structure for {loto} {m}"
        cfg = lp.LOTO_CONFIG[loto]
        lo, hi = cfg["range"]
        pick = cfg["pick"]
        n_pool = hi - lo + 1
        masks = edges_to_masks(edges, m, pick)
        port = shape_polish(masks_to_portfolio(masks, m, n_pool, lo), cfg)
        f_tool = fail_count(portfolio_to_masks(port), m, n_pool, pick)
        f_prod = lp._fail_count_under_threshold(port, loto, 3)
        assert f_tool == f_prod, (loto, m, f_tool, f_prod)
        total = comb(n_pool, pick)
        any3 = 1 - f_prod / total
        print(f"    # fail3={f_prod}, exact any3={any3*100:.4f}%")
        print(f"    ({loto!r}, {m}): (")
        for t in port:
            print(f"        {t},")
        print("    ),")
    print("}")


def main():
    cmd = sys.argv[1]
    if cmd == "family":
        family_search(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "anneal":
        best, port = anneal(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                            float(sys.argv[5]))
        print(json.dumps({"fail3": best, "portfolio": [list(t) for t in port]}))
    elif cmd == "shrink":
        best, port, removed = shrink_next_portfolio(
            sys.argv[2], int(sys.argv[3]), verbose=False,
        )
        loto = sys.argv[2]
        num_sets = int(sys.argv[3])
        current = lp._PRECOMPUTED_BEST_FOUND_PORTFOLIOS.get((loto, num_sets))
        current_fail3 = None
        if current is not None:
            cfg = lp.LOTO_CONFIG[loto]
            lo, hi = cfg["range"]
            current_fail3 = fail_count(
                portfolio_to_masks(current), num_sets, hi - lo + 1, cfg["pick"],
            )
        print(json.dumps({
            "fail3": best,
            "current_fail3": current_fail3,
            "delta": best - current_fail3 if current_fail3 is not None else None,
            "is_improvement": current_fail3 is None or best < current_fail3,
            "removed_index": removed,
            "portfolio": [list(t) for t in port],
        }))
    elif cmd == "retune":
        # retune <loto> <num_sets> [budget_s] [starts]
        print(json.dumps(retune(
            sys.argv[2], int(sys.argv[3]),
            float(sys.argv[4]) if len(sys.argv) > 4 else 150.0,
            int(sys.argv[5]) if len(sys.argv) > 5 else 3,
        )))
    elif cmd == "prize-prob":
        # prize-prob <loto> <num_sets> — 出荷テーブルの exact 入賞確率
        loto, num_sets = sys.argv[2], int(sys.argv[3])
        port = [tuple(t) for t in lp._PRECOMPUTED_BEST_FOUND_PORTFOLIOS[(loto, num_sets)]]
        print(json.dumps({"loto": loto, "num_sets": num_sets,
                          "prize_prob": exact_prize_prob(port, loto)}))
    elif cmd == "emit":
        emit_table()
    elif cmd == "emit-plus":
        emit_plus_table()
    else:
        raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
