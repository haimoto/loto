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
    with pytest.raises(ValueError, match="上限は 10"):
        lp.generate_hitprob_from_draws([], "loto6", num_sets=11)
