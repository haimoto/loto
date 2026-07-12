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
        v5.9-v5.11。全空間の最適性証明なし）
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
    ('loto6', 14): {1: 1, 5: 1, 8: 1, 9: 1, 12: 1, 17: 1, 20: 1, 33: 1, 36: 1, 48: 1, 66: 1, 130: 1, 192: 1, 257: 1, 260: 1, 264: 1, 272: 1, 288: 1, 514: 1, 528: 1, 544: 1, 640: 1, 1026: 1, 1056: 1, 1088: 1, 1152: 1, 1536: 1, 2052: 1, 2056: 1, 2064: 1, 2112: 1, 2304: 1, 4098: 1, 4160: 1, 4224: 1, 4608: 1, 5120: 1, 8194: 1, 8200: 1, 8256: 1, 8320: 1, 10240: 1, 12288: 1},
    ('loto6', 15): {5: 1, 9: 1, 10: 1, 17: 1, 18: 1, 20: 1, 33: 1, 34: 1, 56: 1, 132: 1, 160: 1, 192: 1, 320: 1, 513: 1, 516: 1, 520: 1, 640: 1, 768: 1, 1026: 1, 1040: 1, 1280: 1, 2052: 1, 2112: 1, 2432: 1, 2560: 1, 3072: 1, 4104: 1, 4128: 1, 4160: 1, 4354: 1, 5120: 1, 8194: 1, 8200: 1, 8256: 1, 8448: 1, 9216: 1, 12304: 1, 16385: 1, 16388: 1, 16416: 1, 16448: 1, 16512: 1, 18432: 1},
    ('loto7', 11): {9: 1, 10: 1, 18: 1, 21: 1, 24: 1, 33: 1, 34: 1, 36: 1, 48: 1, 65: 1, 66: 1, 68: 1, 72: 1, 130: 1, 132: 1, 136: 1, 144: 1, 160: 1, 257: 1, 260: 1, 272: 1, 288: 1, 320: 1, 384: 1, 518: 1, 520: 1, 544: 1, 576: 1, 641: 1, 768: 1, 1025: 1, 1026: 1, 1028: 1, 1032: 1, 1040: 1, 1088: 1, 1536: 1},
    ('loto7', 12): {3: 1, 5: 1, 17: 1, 24: 1, 33: 1, 40: 1, 48: 1, 65: 1, 66: 1, 68: 1, 130: 1, 132: 1, 136: 1, 144: 1, 257: 1, 268: 1, 272: 1, 290: 1, 320: 1, 514: 1, 516: 1, 520: 1, 576: 1, 672: 1, 1032: 1, 1042: 1, 1060: 1, 1088: 1, 1153: 1, 1792: 1, 2054: 1, 2056: 1, 2064: 1, 2144: 1, 2432: 1, 2560: 1, 3072: 1},
    ('loto7', 13): {5: 1, 12: 1, 24: 1, 36: 1, 42: 1, 68: 1, 82: 1, 130: 1, 148: 1, 161: 1, 259: 1, 264: 1, 272: 1, 448: 1, 520: 1, 544: 1, 577: 1, 640: 1, 1025: 1, 1026: 1, 1120: 1, 1152: 1, 1280: 1, 1552: 1, 2054: 1, 2065: 1, 2080: 1, 2112: 1, 2184: 1, 2816: 1, 4097: 1, 4104: 1, 4144: 1, 4160: 1, 4356: 1, 4610: 1, 7168: 1},
    ('loto7', 14): {12: 1, 24: 1, 36: 1, 42: 1, 68: 1, 82: 1, 148: 1, 161: 1, 259: 1, 264: 1, 448: 1, 520: 1, 577: 1, 640: 1, 1025: 1, 1120: 1, 1152: 1, 1280: 1, 1552: 1, 2054: 1, 2065: 1, 2080: 1, 2184: 1, 2816: 1, 4101: 1, 4226: 1, 4368: 1, 4640: 1, 5122: 1, 6208: 1, 8193: 1, 8240: 1, 8256: 1, 8452: 1, 8706: 1, 11264: 1, 12296: 1},
    ('loto7', 15): {3: 1, 26: 1, 81: 1, 100: 1, 168: 1, 296: 1, 388: 1, 514: 1, 517: 1, 584: 1, 656: 1, 1025: 1, 1072: 1, 1090: 1, 1792: 1, 2060: 1, 2082: 1, 2240: 1, 2320: 1, 4102: 1, 4225: 1, 4640: 1, 5128: 1, 6400: 1, 8212: 1, 8224: 1, 8512: 1, 9344: 1, 10241: 1, 12352: 1, 16388: 1, 16514: 1, 16641: 1, 17408: 1, 18944: 1, 20496: 1, 24584: 1},
    ('loto6', 16): {6: 1, 25: 1, 33: 1, 48: 1, 66: 1, 129: 1, 160: 1, 258: 1, 260: 1, 384: 1, 516: 1, 520: 1, 528: 1, 608: 1, 640: 1, 1089: 1, 1160: 1, 1280: 1, 2049: 1, 2088: 1, 2128: 1, 2176: 1, 3074: 1, 4108: 1, 4112: 1, 4128: 1, 4352: 1, 8196: 1, 8208: 1, 8264: 1, 8704: 1, 12288: 1, 16385: 1, 16386: 1, 16448: 1, 17408: 1, 26624: 1, 32770: 1, 32772: 1, 33024: 1, 33792: 1, 36864: 1, 49152: 1},
    ('loto6', 17): {3: 1, 9: 1, 42: 1, 129: 1, 132: 1, 192: 1, 257: 1, 264: 1, 322: 1, 520: 1, 528: 1, 1040: 1, 1056: 1, 1152: 1, 1284: 1, 2050: 1, 2112: 1, 4256: 1, 4608: 1, 5122: 1, 6144: 1, 8193: 1, 8198: 1, 8240: 1, 8576: 1, 16385: 1, 16392: 1, 16420: 1, 16448: 1, 16896: 1, 18448: 1, 32776: 1, 32784: 1, 32836: 1, 34848: 1, 36864: 1, 41472: 1, 65552: 1, 65600: 1, 65792: 1, 68096: 1, 69636: 1, 74752: 1},
    ('loto6', 18): {25: 1, 36: 1, 65: 1, 66: 1, 192: 1, 256: 1, 257: 1, 268: 1, 513: 1, 1028: 1, 1032: 1, 1152: 1, 1538: 1, 2050: 1, 3104: 1, 4114: 1, 4352: 1, 4736: 1, 8193: 1, 8208: 1, 8480: 1, 10244: 1, 12288: 1, 16480: 1, 16516: 1, 17408: 1, 18448: 1, 32784: 1, 32800: 1, 32896: 1, 33288: 1, 36928: 1, 65539: 1, 66080: 1, 69640: 1, 82432: 1, 100352: 1, 131078: 1, 131456: 1, 133120: 1, 139328: 1, 147464: 1, 196624: 1},
    ('loto6', 19): {10: 1, 84: 1, 161: 1, 288: 1, 320: 1, 529: 1, 608: 1, 896: 1, 1025: 1, 1028: 1, 2053: 1, 2080: 1, 2176: 1, 3072: 1, 4098: 1, 4168: 1, 4612: 1, 8464: 1, 9248: 1, 17410: 1, 18432: 1, 24576: 1, 32792: 1, 33792: 1, 40961: 1, 53376: 1, 65568: 1, 65600: 1, 65682: 1, 73736: 1, 81921: 1, 131140: 1, 131200: 1, 131330: 1, 131592: 1, 163840: 1, 200704: 1, 262656: 1, 264208: 1, 266496: 1, 270338: 1, 278536: 1, 294916: 1},
    ('loto6', 20): {10: 1, 84: 1, 289: 1, 544: 1, 576: 1, 1041: 1, 1120: 1, 1792: 1, 2049: 1, 2052: 1, 4101: 1, 4128: 1, 4480: 1, 6144: 1, 8264: 1, 8322: 1, 9220: 1, 16912: 1, 18464: 1, 34818: 1, 36864: 1, 49280: 1, 65560: 1, 67584: 1, 81921: 1, 106752: 1, 131136: 1, 131232: 1, 131346: 1, 147464: 1, 163841: 1, 262212: 1, 262400: 1, 262658: 1, 263304: 1, 327680: 1, 401408: 1, 525440: 1, 528400: 1, 532992: 1, 540674: 1, 557064: 1, 589828: 1},
    ('loto7', 16): {37: 1, 40: 1, 67: 1, 88: 1, 416: 1, 524: 1, 608: 1, 1044: 1, 1232: 1, 1282: 1, 1537: 1, 2058: 1, 2113: 1, 2240: 1, 2816: 1, 4144: 1, 4225: 1, 4354: 1, 8322: 1, 8324: 1, 8464: 1, 8705: 1, 9224: 1, 10272: 1, 16644: 1, 16648: 1, 16914: 1, 19456: 1, 20484: 1, 20512: 1, 32777: 1, 32836: 1, 33408: 1, 34832: 1, 37888: 1, 45056: 1, 49154: 1},
    ('loto7', 17): {38: 1, 67: 1, 81: 1, 152: 1, 280: 1, 448: 1, 608: 1, 1038: 1, 1104: 1, 2336: 1, 4144: 1, 4168: 1, 4226: 1, 4356: 1, 7168: 1, 8232: 1, 8450: 1, 8836: 1, 10241: 1, 16912: 1, 17536: 1, 18592: 1, 20481: 1, 24640: 1, 34560: 1, 34819: 1, 34820: 1, 38400: 1, 40960: 1, 49160: 1, 65669: 1, 66050: 1, 66593: 1, 68104: 1, 73744: 1, 81924: 1, 98560: 1},
    ('loto7', 18): {102: 1, 131: 1, 145: 1, 280: 1, 536: 1, 896: 1, 1216: 1, 2062: 1, 2192: 1, 4672: 1, 8272: 1, 8328: 1, 8482: 1, 8708: 1, 14368: 1, 16456: 1, 16898: 1, 17668: 1, 20513: 1, 33808: 1, 35104: 1, 37184: 1, 40961: 1, 49280: 1, 69120: 1, 69635: 1, 69636: 1, 76800: 1, 81952: 1, 98312: 1, 131333: 1, 132098: 1, 133185: 1, 136200: 1, 147472: 1, 163844: 1, 197152: 1},
    ('loto7', 19): {102: 1, 673: 1, 1136: 1, 2058: 1, 2069: 1, 2336: 1, 2624: 1, 4242: 1, 7168: 1, 8472: 1, 9728: 1, 12293: 1, 16705: 1, 17540: 1, 32777: 1, 32960: 1, 33296: 1, 34052: 1, 40992: 1, 51200: 1, 66306: 1, 66569: 1, 69664: 1, 73800: 1, 131280: 1, 139266: 1, 147968: 1, 168192: 1, 197760: 1, 200708: 1, 262668: 1, 264576: 1, 278530: 2, 282632: 1, 335888: 1, 393249: 1},
    ('loto7', 20): {387: 1, 1096: 1, 1568: 1, 2160: 1, 3204: 1, 4146: 1, 4614: 1, 8213: 1, 8328: 1, 16548: 1, 18952: 1, 25856: 1, 37952: 1, 43010: 1, 49218: 1, 65728: 1, 65832: 1, 76288: 1, 77825: 1, 81953: 1, 98312: 1, 131137: 1, 131593: 1, 139536: 1, 263040: 1, 264272: 1, 266252: 1, 311312: 1, 394242: 1, 425988: 1, 525313: 1, 528644: 1, 557328: 1, 590338: 1, 659584: 1, 673792: 1, 786464: 1},
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
    elif cmd == "emit":
        emit_table()
    elif cmd == "emit-plus":
        emit_plus_table()
    else:
        raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
