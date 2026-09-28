"""Culture canonicalization.

Regression cover for the fragmentation that broke the first run: 2,410 distinct
culture strings for 22,990 records, and zero exact overlap between the 45
benchmark culture labels and the pathway culture labels.
"""
import pytest

from src.core.culture_normalizer import (
    canonical_culture,
    canonical_cultures,
    cultures_match,
    is_region,
)


@pytest.mark.parametrize("value,expected", [
    ("American", "united states"),
    ("Americans", "united states"),
    ("USA", "united states"),
    ("United States", "united states"),
    ("Japanese", "japan"),
    ("Japan", "japan"),
    ("Dutch", "netherlands"),
    ("Netherlands", "netherlands"),
    ("Filipinos", "philippines"),
    ("Philippines", "philippines"),
    ("Czech", "czech republic"),
    ("Czech Republic", "czech republic"),
    ("British", "united kingdom"),
    ("UK", "united kingdom"),
    ("Swedish people", "sweden"),
    ("the Germans", "germany"),
    ("Saudi", "saudi arabia"),
    ("Saudi Arabia", "saudi arabia"),
])
def test_canonical_culture(value, expected):
    assert canonical_culture(value) == expected


@pytest.mark.parametrize("a,b", [
    ("Americans", "United States"),
    ("Japanese", "Japan"),
    ("Korean", "South Korea"),
    ("Swedish people", "Sweden"),
    ("Zimbabwean", "Zimbabwe"),
])
def test_country_and_demonym_match(a, b):
    assert cultures_match(a, b)


@pytest.mark.parametrize("a,b", [
    ("European", "France"),      # region must not collapse into a member state
    ("Japanese", "China"),
    ("American", "Australian"),  # substring overlap must not create a match
])
def test_distinct_cultures_do_not_match(a, b):
    assert not cultures_match(a, b)


def test_empty_values_never_match():
    assert not cultures_match("", "Japan")
    assert not cultures_match("Japan", None)
    assert canonical_culture(None) == ""


def test_regions_detected():
    assert is_region("Europeans")
    assert not is_region("Germany")


def test_unrecognized_groups_still_collapse_variants():
    """Free-text groups keep a stable key rather than becoming 'unknown'."""
    a = canonical_culture("people who take things seriously")
    assert a
    assert a == canonical_culture("The people who take things seriously")


def test_canonical_cultures_drops_empties():
    assert canonical_cultures(["Japanese", "", None, "Japan"]) == {"japan"}
