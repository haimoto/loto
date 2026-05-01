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


def test_hitprob_rejects_more_than_max_disjoint_sets():
    with pytest.raises(ValueError, match="最大組数は 7 組"):
        lp.generate_hitprob_from_draws([], "loto6", num_sets=8)
