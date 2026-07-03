import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import loto_predictor_chatgpt as lp


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


# Globally optimal fail3 (count of draws where no ticket reaches 3 hits) for each
# extended portfolio, derived and verified offline by optimize_hitprob_extended.py.
# Lower is better. These lock the v5.8 precomputed table against regression: any
# future edit to a portfolio must not raise fail3 above the proven optimum.
_OPTIMAL_FAIL3 = {
    ("loto6", 8): 4798122,
    ("loto6", 9): 4653802,
    ("loto6", 10): 4509695,
    ("loto7", 6): 4364271,
    ("loto7", 7): 3790435,
    ("loto7", 8): 3223042,
    ("loto7", 9): 2669273,
    ("loto7", 10): 2175649,
}


@pytest.mark.parametrize("loto,num_sets", sorted(_OPTIMAL_FAIL3))
def test_precomputed_portfolio_is_globally_optimal(loto, num_sets):
    sets = _portfolio(loto, num_sets)
    fail3 = lp._fail_count_under_threshold([tuple(s) for s in sets], loto, 3)
    assert fail3 == _OPTIMAL_FAIL3[(loto, num_sets)]


def test_precomputed_beats_v57_disjoint_plus_leftover_hillclimb():
    """The precomputed optimum must be no worse than the old local-optimum seed.

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
    ("loto6", 14): 3938577,
    ("loto6", 15): 3809045,
    ("loto7", 11): 1797292,
    ("loto7", 12): 1543360,
    ("loto7", 13): 1299108,
    ("loto7", 14): 1051869,
    ("loto7", 15): 831014,
    ("loto6", 16): 3684920,
    ("loto6", 17): 3562131,
    ("loto6", 18): 3446997,
    ("loto6", 19): 3336190,
    ("loto6", 20): 3211133,
    ("loto7", 16): 708921,
    ("loto7", 17): 616250,
    ("loto7", 18): 505817,
    ("loto7", 19): 407876,
    ("loto7", 20): 345176,
}


@pytest.mark.parametrize("loto,num_sets", sorted(_BEST_FOUND_FAIL3))
def test_best_found_portfolio_matches_lock(loto, num_sets):
    sets = _portfolio(loto, num_sets)
    fail3 = lp._fail_count_under_threshold([tuple(s) for s in sets], loto, 3)
    assert fail3 == _BEST_FOUND_FAIL3[(loto, num_sets)]


def test_any3_strictly_increases_with_num_sets_up_to_cap():
    """Guaranteed by construction (appending a ticket only shrinks the fail
    set); guards the 11-15 best-found entries against a bad table edit."""
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
