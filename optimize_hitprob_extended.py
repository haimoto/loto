"""Offline optimizer for the extended hitprob portfolios (overlap structures).

`loto_predictor_chatgpt._PRECOMPUTED_EXTENDED_PORTFOLIOS` のテーブルは
このスクリプトで導出した。再現・再検証・上限変更時の再生成に使う。

理論的背景:
  any3 確率（少なくとも1口が3個以上ヒット）は、各数字が「どのチケット集合に
  属するか」(membership mask) の多重集合だけで決まり、具体的な数字には依存
  しない。完全非重複上限を超える組数では合計スロット数 m*pick がプールサイズ
  N を超えるため、超過分 E = m*pick - N をどの構造で重ねるかが唯一の自由度。

探索結果（2026-06-11）:
  - 最適構造は常に「相異なるペアの単純グラフを最小サポート k
    (= min{k: C(k,2) >= E, 次数 <= pick}) のチケット部分集合に密に載せ、
    補グラフが1頂点に集中（K_k マイナス スター）になる形」だった。
  - loto7 6口は全構造空間（マスクサイズ2-6, 未使用数字0-2）の全数探索で、
    loto6 8口はマスクサイズ2-3・未使用数字0-1 の全数探索で厳密最適を確認。
  - 他の組数は dense-family 全数探索 + 多スタート焼きなましの一致で確認
   （独立シードがすべて同一値に収束）。

使い方:
  python3 optimize_hitprob_extended.py family <loto> <num_sets> <k>
  python3 optimize_hitprob_extended.py anneal <loto> <num_sets> <seed> <minutes>
  python3 optimize_hitprob_extended.py emit
      （既知の最良構造から確定テーブルを再構築し、本番DPで fail3 を検証して
        Python リテラルを出力する）
  python3 optimize_hitprob_extended.py emit-plus
      （11-15口の best-found 構造 BEST_PLUS_MASKS から
        _PRECOMPUTED_BEST_FOUND_PORTFOLIOS を再構築・検証・出力する。v5.9。
        ペアのみの重複が次数制約で不可能な領域のため最適性証明なし）
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

    port = [list(s) for s in best_port]
    improved = True
    while improved and time.time() < t_end:
        improved = False
        for i in range(m):
            for j in range(pick):
                old = port[i][j]
                others = set(port[i]) - {old}
                for v in pool:
                    if v == old or v in others:
                        continue
                    port[i][j] = v
                    cand = evaluate(port)
                    if cand < best:
                        best, old, improved = cand, v, True
                    else:
                        port[i][j] = old
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
    # Densest case (E=33 on 10 tickets, support must be all 10). The optimum is
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


# Best-found mask structures for 11-15 sets (拡張プラスモード, v5.9). Pair-only
# simple graphs are infeasible here: excess E = m*pick - N exceeds the max edge
# count of a degree<=pick graph, so numbers must be shared by 3+ tickets and
# the structure is stored as {membership mask: count} over ticket indices.
# Derived by multi-start simulated annealing (2 seeds per case, best kept).
# NO optimality proof, unlike BEST_EXCESS_EDGES above.
BEST_PLUS_MASKS = {
    ('loto6', 11): {1: 5, 6: 1, 10: 1, 12: 1, 20: 1, 24: 1, 34: 1, 36: 1, 40: 1, 48: 1, 66: 1, 68: 1, 72: 1, 80: 1, 96: 1, 128: 6, 256: 5, 257: 1, 512: 4, 514: 1, 528: 1, 1026: 1, 1028: 1, 1032: 1, 1040: 1, 1056: 1, 1088: 1},
    ('loto6', 12): {2: 3, 4: 1, 5: 1, 8: 2, 10: 1, 16: 3, 24: 1, 32: 2, 34: 1, 40: 1, 48: 1, 65: 1, 68: 1, 129: 1, 132: 1, 192: 1, 256: 2, 258: 1, 264: 1, 272: 1, 288: 1, 512: 1, 513: 1, 576: 1, 640: 1, 1025: 1, 1028: 1, 1088: 1, 1152: 1, 1536: 1, 2049: 1, 2052: 1, 2112: 1, 2176: 1, 2560: 1, 3072: 1},
    ('loto6', 13): {1: 1, 2: 1, 3: 1, 8: 2, 9: 1, 10: 1, 20: 1, 36: 1, 48: 1, 68: 1, 80: 1, 96: 1, 128: 2, 129: 1, 130: 1, 260: 1, 272: 1, 288: 1, 320: 1, 512: 1, 513: 1, 514: 1, 520: 1, 640: 1, 1024: 1, 1025: 1, 1026: 1, 1032: 1, 1152: 1, 1536: 1, 2052: 1, 2064: 1, 2080: 1, 2112: 1, 2304: 1, 4100: 1, 4112: 1, 4128: 1, 4160: 1, 4352: 1, 6144: 1},
    ('loto6', 14): {1: 1, 5: 1, 8: 1, 9: 1, 12: 1, 17: 1, 20: 1, 33: 1, 36: 1, 48: 1, 66: 1, 130: 1, 192: 1, 257: 1, 260: 1, 264: 1, 272: 1, 288: 1, 514: 1, 528: 1, 544: 1, 640: 1, 1026: 1, 1056: 1, 1088: 1, 1152: 1, 1536: 1, 2052: 1, 2056: 1, 2064: 1, 2112: 1, 2304: 1, 4098: 1, 4160: 1, 4224: 1, 4608: 1, 5120: 1, 8194: 1, 8200: 1, 8256: 1, 8320: 1, 10240: 1, 12288: 1},
    ('loto6', 15): {5: 1, 9: 1, 10: 1, 17: 1, 18: 1, 20: 1, 33: 1, 34: 1, 56: 1, 132: 1, 160: 1, 192: 1, 320: 1, 513: 1, 516: 1, 520: 1, 640: 1, 768: 1, 1026: 1, 1040: 1, 1280: 1, 2052: 1, 2112: 1, 2432: 1, 2560: 1, 3072: 1, 4104: 1, 4128: 1, 4160: 1, 4354: 1, 5120: 1, 8194: 1, 8200: 1, 8256: 1, 8448: 1, 9216: 1, 12304: 1, 16385: 1, 16388: 1, 16416: 1, 16448: 1, 16512: 1, 18432: 1},
    ('loto7', 11): {3: 1, 6: 1, 9: 1, 12: 1, 20: 1, 24: 1, 33: 1, 34: 1, 40: 1, 65: 1, 66: 1, 68: 1, 80: 1, 129: 1, 136: 1, 144: 1, 160: 1, 192: 1, 258: 1, 264: 1, 272: 1, 288: 1, 320: 1, 388: 1, 513: 1, 514: 1, 516: 1, 528: 1, 544: 1, 576: 1, 1026: 1, 1029: 1, 1032: 1, 1040: 1, 1056: 1, 1152: 1, 1792: 1},
    ('loto7', 12): {3: 1, 5: 1, 17: 1, 24: 1, 33: 1, 40: 1, 48: 1, 65: 1, 66: 1, 68: 1, 130: 1, 132: 1, 136: 1, 144: 1, 257: 1, 268: 1, 272: 1, 290: 1, 320: 1, 514: 1, 516: 1, 520: 1, 576: 1, 672: 1, 1032: 1, 1042: 1, 1060: 1, 1088: 1, 1153: 1, 1792: 1, 2054: 1, 2056: 1, 2064: 1, 2144: 1, 2432: 1, 2560: 1, 3072: 1},
    ('loto7', 13): {3: 1, 9: 1, 28: 1, 38: 1, 40: 1, 74: 1, 80: 1, 145: 1, 196: 1, 264: 1, 352: 1, 386: 1, 513: 1, 514: 1, 532: 1, 544: 1, 576: 1, 640: 1, 768: 1, 1029: 1, 1072: 1, 1088: 1, 1152: 1, 1280: 1, 2049: 1, 2066: 1, 2080: 1, 2184: 1, 2308: 1, 3072: 1, 4097: 1, 4108: 1, 4128: 1, 4224: 1, 4368: 1, 5122: 1, 6208: 1},
    ('loto7', 14): {12: 1, 24: 1, 36: 1, 42: 1, 68: 1, 82: 1, 148: 1, 161: 1, 259: 1, 264: 1, 448: 1, 520: 1, 577: 1, 640: 1, 1025: 1, 1120: 1, 1152: 1, 1280: 1, 1552: 1, 2054: 1, 2065: 1, 2080: 1, 2184: 1, 2816: 1, 4101: 1, 4226: 1, 4368: 1, 4640: 1, 5122: 1, 6208: 1, 8193: 1, 8240: 1, 8256: 1, 8452: 1, 8706: 1, 11264: 1, 12296: 1},
    ('loto7', 15): {3: 1, 26: 1, 81: 1, 100: 1, 168: 1, 296: 1, 388: 1, 514: 1, 517: 1, 584: 1, 656: 1, 1025: 1, 1072: 1, 1090: 1, 1792: 1, 2060: 1, 2082: 1, 2240: 1, 2320: 1, 4102: 1, 4225: 1, 4640: 1, 5128: 1, 6400: 1, 8212: 1, 8224: 1, 8512: 1, 9344: 1, 10241: 1, 12352: 1, 16388: 1, 16514: 1, 16641: 1, 17408: 1, 18944: 1, 20496: 1, 24584: 1},
}


def emit_plus_table():
    """Rebuild the best-found 11-15 set portfolios from BEST_PLUS_MASKS,
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
    elif cmd == "emit":
        emit_table()
    elif cmd == "emit-plus":
        emit_plus_table()
    else:
        raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
