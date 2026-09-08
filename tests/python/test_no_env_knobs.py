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


@pytest.mark.parametrize("tier,setter,prop,value,default", [
    ("public", "set_sleeping", "sleeping", False, True),
    ("public", "set_verlet_skin", "verlet_skin", 0.3, 0.0),
    ("public", "set_incremental_coloring", "incremental_coloring", False, True),
    # bit-identical execution policies live on the diagnostics tier (QUALITY_PLAN D2)
    ("diagnostics", "set_cuda_graphs", "cuda_graphs", False, True),
    ("diagnostics", "set_fused_sweeps", "fused_sweeps", "off", "auto"),
])
def test_execution_policy_setters_round_trip(tier, setter, prop, value, default):
    from peclet import dem
    s = dem.Simulation(8)
    obj = s.diagnostics if tier == "diagnostics" else s
    approx = pytest.approx if isinstance(value, float) else (lambda x: x)
    assert getattr(obj, prop) == approx(default)
    getattr(obj, setter)(value)
    assert getattr(obj, prop) == approx(value)
    assert not hasattr(s.diagnostics if tier == "public" else s, setter), (
        f"{setter} is bound on both tiers")
