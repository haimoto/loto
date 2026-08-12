"""takarakuji_fill の Safari 非依存部分のテスト。

AppleScript / Safari を叩く部分（find_tab / fill / verify）は実環境依存なので対象外。
ここで守るのは「生成した組が妥当か」「JS へ正しく埋め込まれるか」の2点。
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import takarakuji_fill as tf  # noqa: E402


def test_parse_sets_accepts_space_and_comma():
    assert tf.parse_sets("1 2 3 4 5 6 7/8,9,10,11,12,13,14") == [
        [1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14],
    ]


def test_parse_sets_sorts_each_set():
    assert tf.parse_sets("7 1 5 3 2 6 4") == [[1, 2, 3, 4, 5, 6, 7]]


@pytest.mark.parametrize(
    "sets, loto",
    [
        ([[1, 2, 3, 4, 5, 6]], "loto7"),          # 個数不足
        ([[1, 2, 3, 4, 5, 6, 6]], "loto7"),       # 重複
        ([[1, 2, 3, 4, 5, 6, 38]], "loto7"),      # 範囲外
        ([[1, 2, 3, 4, 5, 44]], "loto6"),         # 範囲外
        ([[1, 2, 3, 4, 5]], "loto6"),             # 個数不足
    ],
)
def test_validate_rejects_bad_sets(sets, loto):
    with pytest.raises(SystemExit):
        tf.validate(sets, loto)


def test_validate_accepts_good_sets():
    tf.validate([[1, 2, 3, 4, 5, 6, 37]], "loto7")
    tf.validate([[1, 2, 3, 4, 5, 43]], "loto6")


def test_js_literal_escapes_backslash_and_quote():
    assert tf._js_literal('a\\b"c') == '"a\\\\b\\"c"'


def test_select_js_embeds_numbers_and_dispatches_mouse_events():
    js = tf._select_js([1, 2, 3, 4, 5, 6, 7])
    assert "var want=[1,2,3,4,5,6,7]" in js
    for ev in ("mouseover", "mousedown", "mouseup", "click"):
        assert ev in js
    # JS 側は AppleScript 文字列に埋めるためダブルクォート禁止
    assert '"' not in js


def test_advance_and_verify_js_have_no_double_quotes():
    assert '"' not in tf._advance_js()
    assert '"' not in tf._verify_js()


@pytest.mark.parametrize("loto, pick", [("loto6", 6), ("loto7", 7)])
def test_build_sets_shape(loto, pick):
    sets = tf.build_sets(loto, hitprob=3, ev=2, csv_path=REPO / f"{loto}_data.csv")
    assert len(sets) == 5
    tf.validate(sets, loto)
    assert all(len(s) == pick and s == sorted(s) for s in sets)


def test_build_sets_hitprob_is_history_independent():
    """hitprob は履歴非依存＝CSV が伸びても同じ組が出る（回号違いで別数字を期待しない）。"""
    a = tf.build_sets("loto7", hitprob=5, ev=0, csv_path=REPO / "loto7_data.csv")
    b = tf.build_sets("loto7", hitprob=5, ev=0, csv_path=REPO / "loto7_data.csv")
    assert a == b
    assert len({tuple(s) for s in a}) == 5


def test_build_sets_zero_counts():
    assert tf.build_sets("loto7", hitprob=0, ev=0, csv_path=REPO / "loto7_data.csv") == []
