import json
import shutil
import subprocess
from pathlib import Path

import pytest


NODE = shutil.which("node")
POSTERIOR_JS = Path(__file__).resolve().parents[1] / "dashboard" / "posterior.js"


def run_posterior_javascript(expression):
    if NODE is None:
        pytest.skip("Node.js is required for dashboard numerical tests")
    script = f"const p = require({json.dumps(str(POSTERIOR_JS))}); console.log(JSON.stringify({expression}));"
    result = subprocess.run(
        [NODE, "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_interpolation_zero_pads_outside_model_support_and_normalizes():
    result = run_posterior_javascript(
        "(() => { const g=[0,1,2,3,4]; const y=p.interpolatePDF([1,2,3],[0,1,0],g); "
        "return {y, area:p.integrate(g,y)}; })()"
    )

    assert result["y"][0] == 0
    assert result["y"][4] == 0
    assert result["area"] == pytest.approx(1)


def test_joint_posterior_is_normalized_product():
    result = run_posterior_javascript(
        "(() => { const g=[0,1,2,3,4]; const a=p.normalize(g,[0,1,2,1,0]); "
        "const b=p.normalize(g,[0,0,1,2,1]); const j=p.combineIndependent(g,a,b); "
        "return {status:j.status, pdf:j.pdf, area:p.integrate(g,j.pdf), stats:j.stats}; })()"
    )

    assert result["status"] == "ok"
    assert result["area"] == pytest.approx(1)
    assert result["pdf"][0] == 0
    assert result["pdf"][4] == 0
    assert 1.5 < result["stats"]["median"] < 3


def test_non_overlapping_posteriors_do_not_create_a_joint_age():
    result = run_posterior_javascript(
        "p.combineIndependent([0,1,2,3],[1,1,0,0],[0,0,1,1])"
    )

    assert result == {"pdf": None, "stats": None, "status": "no-overlap"}


def test_probability_below_returns_fraction_below_threshold():
    result = run_posterior_javascript(
        "(() => { const g=[0,10,20,30,40,50]; const y=p.normalize(g,[0,0,1,1,0,0]); "
        "return {below25:p.probabilityBelow(g,y,25), below35:p.probabilityBelow(g,y,35), below5:p.probabilityBelow(g,y,5)}; })()"
    )

    assert result["below25"] == pytest.approx(0.5, abs=0.01)
    assert result["below35"] == pytest.approx(0.9375, abs=0.01)
    assert result["below5"] == pytest.approx(0.0, abs=0.01)


def test_probability_below_handles_degenerate_inputs():
    result = run_posterior_javascript(
        "(() => { const g=[0,10,20]; const y=[0,0,0]; "
        "return p.probabilityBelow(g,y,15); })()"
    )

    assert result == 0
