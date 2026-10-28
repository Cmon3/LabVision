"""Tests for bench/datasets.py using fake tmp_path filenames only."""

import pytest

from bench.datasets import (
    list_agar,
    list_bbbc005,
    lookup_count,
    matches_plane,
    parse_bbbc005_count,
)


def _touch(tmp_path, *names):
    for name in names:
        (tmp_path / name).touch()
    return tmp_path


def test_parse_bbbc005_count_patterns():
    assert parse_bbbc005_count("plate_count300.tif") == 300
    assert parse_bbbc005_count("plate_Count_42.png") == 42
    assert parse_bbbc005_count("plate_17cells.tif") == 17
    assert parse_bbbc005_count("plate_c8.tif") == 8
    assert parse_bbbc005_count("SimCep_A01_C1_F1_s05_w1_274.tif") == 274


def test_parse_bbbc005_count_missing_raises():
    with pytest.raises(ValueError):
        parse_bbbc005_count("SimCep_A01_C1_F1_s05_w1.tif")


def test_matches_plane():
    assert matches_plane("SimCep_A01_C1_F1_s05.tif")
    assert not matches_plane("SimCep_A01_C1_F2_s05.tif")
    assert not matches_plane("SimCep_A01_C1_F10_s05.tif")
    assert matches_plane("x_F2_y.tif", plane="F2")


def test_list_bbbc005_keeps_f1_only(tmp_path):
    _touch(
        tmp_path,
        "a_F1_count10.tif",
        "b_F1_count20.tif",
        "c_F2_count30.tif",
        "notes.txt",
    )
    samples = list_bbbc005(tmp_path)
    assert [(s.path.name, s.count) for s in samples] == [
        ("a_F1_count10.tif", 10),
        ("b_F1_count20.tif", 20),
    ]


def test_list_bbbc005_bad_filename_raises(tmp_path):
    _touch(tmp_path, "a_F1_nocount.tif")
    with pytest.raises(ValueError):
        list_bbbc005(tmp_path)


def test_list_bbbc005_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_bbbc005(tmp_path / "nope")


def test_list_agar_without_lookup(tmp_path):
    _touch(tmp_path, "dish_a.jpg", "dish_b.png", "readme.md")
    samples = list_agar(tmp_path)
    assert [s.path.name for s in samples] == ["dish_a.jpg", "dish_b.png"]
    assert all(s.count is None for s in samples)


def test_list_agar_with_lookup_by_name_and_stem(tmp_path):
    _touch(tmp_path, "dish_a.jpg", "dish_b.png", "dish_c.png")
    samples = list_agar(
        tmp_path, counts={"dish_a.jpg": 5, "dish_b": 7}
    )
    assert [(s.path.name, s.count) for s in samples] == [
        ("dish_a.jpg", 5),
        ("dish_b.png", 7),
        ("dish_c.png", None),
    ]


def test_lookup_count_prefers_full_name():
    counts = {"dish_a.jpg": 5, "dish_a": 9}
    assert lookup_count("dish_a.jpg", counts) == 5
    assert lookup_count("other.jpg", counts) is None
    assert lookup_count("other.jpg", None) is None
