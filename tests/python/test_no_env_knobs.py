"""Contract: no environment variable changes what peclet.dem computes (QUALITY_PLAN D3).

Every `PECLET_DEM_*` runtime knob was retired at 1.0.0 into a `Simulation` setter (or deleted
with the ablation it served) -- see dem/CLAUDE.md for the table. This test greps `src/` so that
re-introducing a `getenv` is a failing test, not a silent behaviour switch. `*_PROFILE` /
`*_DEBUG` / `*_VERBOSE` / `*_TIMEOUT` reads are instrumentation and stay allowed.
"""
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
ALLOWED_SUFFIXES = ("_PROFILE", "_DEBUG", "_VERBOSE", "_TIMEOUT")
GETENV = re.compile(r'getenv\s*\(\s*"([^"]*)"')


def test_src_reads_no_numerics_env_vars():
    assert SRC.is_dir(), SRC
    offenders = {}
    for f in sorted(list(SRC.glob("*.hpp")) + list(SRC.glob("*.cpp"))):
        for name in GETENV.findall(f.read_text()):
            if not name.endswith(ALLOWED_SUFFIXES):
                offenders.setdefault(f.name, []).append(name)
    assert not offenders, (
        f"getenv of a non-instrumentation variable in src/: {offenders}. "
        "A knob that selects an algorithm or changes a result must be a Simulation setter "
        "(QUALITY_PLAN D3); see the table in dem/CLAUDE.md.")


@pytest.mark.parametrize("setter,prop,value,default", [
    ("set_sleeping", "sleeping", False, True),
    ("set_verlet_skin", "verlet_skin", 0.3, 0.0),
    ("set_cuda_graphs", "cuda_graphs", False, True),
    ("set_fused_sweeps", "fused_sweeps", "off", "auto"),
    ("set_incremental_coloring", "incremental_coloring", False, True),
])
def test_execution_policy_setters_round_trip(setter, prop, value, default):
    from peclet import dem
    s = dem.Simulation(8)
    approx = pytest.approx if isinstance(value, float) else (lambda x: x)
    assert getattr(s, prop) == approx(default)
    getattr(s, setter)(value)
    assert getattr(s, prop) == approx(value)
