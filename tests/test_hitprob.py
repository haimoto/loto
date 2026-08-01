import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import loto_predictor_chatgpt as lp
import optimize_hitprob_extended as opt


def _portfolio(loto, num_sets):
    _, gen, _ = lp.generate_hitprob_from_draws([], loto, num_sets=num_sets)
    return [nums for _, nums in gen.sets]


def test_loto6_seven_disjoint_sets_raise_any3_probability_over_five_sets():
    five = lp.exact_hitprob(_portfolio("loto6", 5), "loto6")
    seven = lp.exact_hitprob(_portfolio("loto6", 7), "loto6")

    assert five["union_size"] == 30
    assert seven["union_size"] == 42
    assert five["avg_pair_overlap"] == 0
    assert seven["avg_pair_overlap"] == 0
    assert seven["any3"] > five["any3"]


def test_extended_loto6_eight_sets_beat_seven_disjoint():
    seven = lp.exact_hitprob(_portfolio("loto6", 7), "loto6")
    eight = lp.exact_hitprob(_portfolio("loto6", 8), "loto6")

    assert eight["any3"] > seven["any3"]


def test_extended_loto7_six_sets_beat_five_disjoint():
    five = lp.exact_hitprob(_portfolio("loto7", 5), "loto7")
    six = lp.exact_hitprob(_portfolio("loto7", 6), "loto7")

    assert five["avg_pair_overlap"] == 0
    assert six["any3"] > five["any3"]


def test_extended_portfolio_is_deterministic():
    assert _portfolio("loto7", 6) == _portfolio("loto7", 6)


def test_extended_portfolio_sets_are_valid():
    cfg = lp.LOTO_CONFIG["loto7"]
    lo, hi = cfg["range"]
    for nums in _portfolio("loto7", 6):
        assert len(nums) == cfg["pick"]
        assert len(set(nums)) == cfg["pick"]
        assert all(lo <= n <= hi for n in nums)
        assert tuple(nums) == tuple(sorted(nums))


def test_hitprob_rejects_more_than_extended_cap():
    with pytest.raises(ValueError, match="上限は 20"):
        lp.generate_hitprob_from_draws([], "loto6", num_sets=21)


# Best-known fail3 (count of draws where no ticket reaches 3 hits) for each
# extended portfolio, derived and verified offline by optimize_hitprob_extended.py.
# The smallest two cases have recorded offline exhaustive checks; the rest are
# best-found.
_BEST_KNOWN_FAIL3 = {
    ("loto6", 8): 4798122,
    ("loto6", 9): 4653802,
    ("loto6", 10): 4509695,
    ("loto7", 6): 4364271,
    ("loto7", 7): 3790435,
    ("loto7", 8): 3223042,
    ("loto7", 9): 2669273,
    ("loto7", 10): 2175649,
}


@pytest.mark.parametrize("loto,num_sets", sorted(_BEST_KNOWN_FAIL3))
def test_precomputed_portfolio_matches_best_known_lock(loto, num_sets):
    sets = _portfolio(loto, num_sets)
    fail3 = lp._fail_count_under_threshold([tuple(s) for s in sets], loto, 3)
    assert fail3 == _BEST_KNOWN_FAIL3[(loto, num_sets)]


def test_precomputed_beats_v57_disjoint_plus_leftover_hillclimb():
    """The precomputed result must be no worse than the old local-search seed.

    Reconstructs the v5.7 starting portfolio (disjoint base + leftover-filled
    extras, no climb) and checks the shipped table reaches at least its any3.
    """
    for loto in ("loto6", "loto7"):
        max_d = lp._max_disjoint_sets(loto)
        cfg = lp.LOTO_CONFIG[loto]
        lo, hi = cfg["range"]
        pick = cfg["pick"]
        pool = list(range(lo, hi + 1))
        for num_sets in range(max_d + 1, lp.HITPROB_MAX_SETS + 1):
            base = [list(d["nums"]) for d in
                    lp._balanced_disjoint_portfolio(loto, num_sets=max_d)]
            leftovers = sorted(set(pool) - set().union(*(set(s) for s in base)))
            n_extra = num_sets - max_d
            extras = [[] for _ in range(n_extra)]
            for idx, n in enumerate(leftovers):
                extras[idx % n_extra].append(n)
            for e, nums in enumerate(extras):
                cands = [n for n in pool if n not in nums]
                nums.extend(lp._select_evenly(cands, pick - len(nums) + e)[e:])
            seed = base + [sorted(s) for s in extras]
            seed_fail3 = lp._fail_count_under_threshold(
                [tuple(s) for s in seed], loto, 3)
            shipped = lp._fail_count_under_threshold(
                [tuple(s) for s in _portfolio(loto, num_sets)], loto, 3)
            assert shipped <= seed_fail3, (loto, num_sets, shipped, seed_fail3)


# Best-found fail3 for the 11-15 set entries (拡張プラスモード, v5.9). These are
# NOT proven optima — they lock the shipped table against regression: any future
# edit must not raise fail3 above the best structure found so far.
_BEST_FOUND_FAIL3 = {
    ("loto6", 11): 4365806,
    ("loto6", 12): 4223057,
    ("loto6", 13): 4079965,
    ("loto6", 14): 3937272,
    ("loto6", 15): 3808860,
    ("loto7", 11): 1797292,
    ("loto7", 12): 1537623,
    ("loto7", 13): 1286253,
    ("loto7", 14): 1042922,
    ("loto7", 15): 809567,
    ("loto6", 16): 3682945,
    ("loto6", 17): 3559713,
    ("loto6", 18): 3434680,
    ("loto6", 19): 3312775,
    ("loto6", 20): 3190362,
    ("loto7", 16): 624359,
    ("loto7", 17): 533887,
    ("loto7", 18): 439570,
    ("loto7", 19): 352783,
    ("loto7", 20): 271446,
}


@pytest.mark.parametrize("loto,num_sets", sorted(_BEST_FOUND_FAIL3))
def test_best_found_portfolio_matches_lock(loto, num_sets):
    sets = _portfolio(loto, num_sets)
    fail3 = lp._fail_count_under_threshold([tuple(s) for s in sets], loto, 3)
    assert fail3 == _BEST_FOUND_FAIL3[(loto, num_sets)]


# v5.11 は 11-20口の一部を「隣接する大きい best-found から1口削除」して導出したため、
# 「N口テーブル == N+1口テーブルの縮約」という関係を test_shrunk_next_size_source_
# matches_updated_mask_table で固定していた。v5.12 で全組数を独立に再探索した結果、
# 縮約より良い構造が見つかりその関係は失われたので、当該テストは削除した。
# shrink_next_portfolio 自体は導出プリミティブとして有用なので残し、出荷テーブルとの
# 一致を要求していた assert だけを外す。
def test_shrink_next_portfolio_returns_true_minimum():
    """縮約は「全1口削除候補の最小」を正しく返すが、独立再探索の出荷値には届かない。

    返り値の整合（candidate が実際に removed を抜いたもので、fail3 がその実測値）まで
    見ないと、誤った fail3 を返す実装でも「出荷以上」の緩い assert は通ってしまう。
    """
    fail3, candidate, removed = opt.shrink_next_portfolio(
        "loto7", 13, verbose=False,
    )
    source = [tuple(t) for t in _portfolio("loto7", 14)]

    assert 0 <= removed < 14
    assert [tuple(t) for t in candidate] == source[:removed] + source[removed + 1:]
    assert fail3 == lp._fail_count_under_threshold(candidate, "loto7", 3)

    best = min(
        lp._fail_count_under_threshold(source[:i] + source[i + 1:], "loto7", 3)
        for i in range(14)
    )
    assert fail3 == best, "全削除候補の最小を返していない"
    assert fail3 >= _BEST_FOUND_FAIL3[("loto7", 13)]


@pytest.mark.parametrize(
    "a,b",
    [(a, b) for a in range(8) for b in range(3) if a + b <= 7],
)
def test_prize_condition_is_equivalent_to_classify_prize(a, b):
    """入賞条件 `a>=3 かつ a+b>=4` が classify_prize と同値であることを全 (a,b) で確認。

    exact_prize_prob はこの同値変形の上に建っているので、ここが崩れると
    入賞確率の値そのものが無意味になる。数え上げ機構ではなく条件式の検証。
    """
    ticket = tuple(range(1, 8))
    # 本数字: ticket 内から a 個 + ticket 外から 7-a 個
    main = list(range(1, a + 1)) + list(range(20, 20 + (7 - a)))
    # ボーナス: ticket 内の残りから b 個 + ticket 外から 2-b 個（main と重ならない帯）
    bonus = list(range(a + 1, a + 1 + b)) + list(range(30, 30 + (2 - b)))
    assert len(main) == 7 and len(bonus) == 2
    assert not (set(main) & set(bonus)), "本数字とボーナスが重複している"
    assert len(set(ticket) & set(main)) == a
    assert len(set(ticket) & set(bonus)) == b

    draw = lp.Draw(number=0, date="", main=tuple(sorted(main)), bonus=tuple(sorted(bonus)))
    won = lp.classify_prize(ticket, draw, "loto7") is not None
    assert won == (a >= 3 and a + b >= 4), (a, b, won)


def test_exact_prize_prob_matches_independent_enumeration():
    """exact 入賞確率の閉形式を、独立な数え上げ機構と突き合わせる。

    loto7 は C(37,7)*C(30,2) が 44 億通りで全列挙できないため、完全非重複 5口
    （= 閉形式の u=0 と u>0 の両方が出る構造）について、本数字とボーナスを
    同時に分配する別解法で数え、値が一致することを見る。
    勝ち条件そのものの正しさは
    test_prize_condition_is_equivalent_to_classify_prize が担保する。
    """
    pytest.importorskip("numpy")  # exact_prize_prob は numpy を使う（導出ツール側のみ）
    from math import comb as _c

    port = [tuple(range(1 + 7 * k, 8 + 7 * k)) for k in range(5)]
    closed = opt.exact_prize_prob(port, "loto7")

    # 別解法: グループ（各口7個 + 未使用2個）へ本数字7個・ボーナス2個を同時に分配
    sizes = [7] * 5 + [2]
    total = _c(37, 7) * _c(30, 2)
    win = 0
    checked = 0

    def rec(g, xs, ys, rx, ry, ways):
        nonlocal win, checked
        if g == len(sizes):
            if rx or ry:
                return
            checked += ways
            if any(xs[i] >= 3 and xs[i] + ys[i] >= 4 for i in range(5)):
                win += ways
            return
        c = sizes[g]
        for x in range(min(c, rx) + 1):
            for y in range(min(c - x, ry) + 1):
                rec(g + 1, xs + [x], ys + [y], rx - x, ry - y,
                    ways * _c(c, x) * _c(c - x, y))

    rec(0, [], [], 7, 2, 1)
    assert checked == total, "全列挙の検算が合わない"
    assert abs(closed - win / total) < 1e-12


def test_shipped_tables_pass_the_prize_probability_gate():
    """v5.12 の採用ゲート: 更新した loto7 の構造は入賞確率も非悪化していること。

    any3 と入賞確率は P(win)=P(any3)*P(win|any3) の関係で、条件付き側が下がれば
    any3 改善でも入賞確率は悪化しうる。テーブル更新時にその向きを固定する。
    """
    pytest.importorskip("numpy")  # exact_prize_prob は numpy を使う（導出ツール側のみ）
    # exact 列挙は1ケース約1分かかるので、改善幅が最大の 20口だけを固定する。
    # 出荷前は全組数で optimize_hitprob_extended.py prize-prob を回して確認すること。
    port = [tuple(t) for t in lp._PRECOMPUTED_BEST_FOUND_PORTFOLIOS[("loto7", 20)]]
    assert opt.exact_prize_prob(port, "loto7") >= 0.6192


def test_any3_strictly_increases_with_num_sets_up_to_cap():
    """Guard independently optimized best-found entries against regression."""
    for loto in ("loto6", "loto7"):
        prev = None
        for num_sets in range(10, lp.HITPROB_MAX_SETS + 1):
            fail3 = lp._fail_count_under_threshold(
                [tuple(s) for s in _portfolio(loto, num_sets)], loto, 3)
            if prev is not None:
                assert fail3 < prev, (loto, num_sets, fail3, prev)
            prev = fail3


def test_extended_generation_is_instant():
    """Regression guard: v5.8 replaced the minutes-long runtime hill climb with a
    table lookup, so the slowest extended size must generate well under a second."""
    import time

    t0 = time.time()
    lp.generate_hitprob_from_draws([], "loto6", num_sets=10)
    lp.generate_hitprob_from_draws([], "loto6", num_sets=20)
    lp.generate_hitprob_from_draws([], "loto7", num_sets=20)
    assert time.time() - t0 < 1.0
