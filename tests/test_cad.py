"""Outline to printable part. Host-side; needs the cad extra (CadQuery)."""

import hashlib
import struct
from pathlib import Path

import pytest

pytest.importorskip("cadquery")

from xfoil_mcp.cad import CadError, build_section

FIXTURES = Path(__file__).parent / "fixtures"


def outline(name: str) -> list[tuple[float, float]]:
    return [tuple(float(v) for v in line.split())
            for line in (FIXTURES / name).read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def clean():
    return build_section(outline("naca2412_flap_none.dat"))


@pytest.fixture(scope="module")
def flapped():
    return build_section(outline("naca2412_flap_10.dat"))


def test_clean_section_has_the_expected_size(clean):
    assert clean.stats["chord_mm"] == pytest.approx(150.0, abs=0.05)
    assert clean.stats["thickness_mm"] == pytest.approx(18.3, abs=0.1)
    assert clean.stats["span_mm"] == pytest.approx(40.0)
    assert clean.stats["volume_cm3"] == pytest.approx(74.0, abs=0.1)


def test_flap_shortens_the_projected_chord_and_drops_the_tail(flapped):
    assert flapped.stats["chord_mm"] == pytest.approx(149.3, abs=0.1)
    assert flapped.stats["thickness_mm"] == pytest.approx(19.9, abs=0.1)


def test_stl_is_a_well_formed_binary_file(clean):
    """Binary STL: an 80-byte header, a triangle count, then 50 bytes per triangle."""
    (count,) = struct.unpack("<I", clean.stl[80:84])
    assert count > 0
    assert len(clean.stl) == 84 + 50 * count


def test_hash_is_of_the_exact_bytes(clean):
    assert clean.sha256 == hashlib.sha256(clean.stl).hexdigest()


def test_same_outline_gives_the_same_file():
    """The approval binds to the hash, so the same design must always hash the same."""
    a = build_section(outline("naca2412_flap_10.dat"))
    b = build_section(outline("naca2412_flap_10.dat"))
    assert a.sha256 == b.sha256


def test_estimates_match_the_real_slice(clean):
    """Calibrated from Bambu Studio: two of these parts took 55.27 g and 64 min."""
    assert clean.stats["estimated_mass_g"] == pytest.approx(27.6, abs=0.5)
    assert clean.stats["estimated_print_minutes"] == pytest.approx(39, abs=2)


def test_requested_dimensions_are_the_part_dimensions():
    part = build_section(outline("naca2412_flap_none.dat"), chord_mm=100.0, span_mm=20.0)
    assert part.stats["chord_mm"] == pytest.approx(100.0, abs=0.05)
    assert part.stats["span_mm"] == pytest.approx(20.0, abs=0.01)


@pytest.mark.parametrize("size", [
    {"chord_mm": 0.0}, {"chord_mm": -5.0}, {"chord_mm": 251.0},
    {"span_mm": 0.5}, {"span_mm": 1000.0}, {"span_mm": float("nan")},
])
def test_dimensions_the_printer_cannot_hold_are_an_error(size):
    with pytest.raises(CadError):
        build_section(outline("naca2412_flap_none.dat"), **size)


def test_too_few_points_is_an_error():
    with pytest.raises(CadError):
        build_section([(0.0, 0.0), (1.0, 0.0)])
