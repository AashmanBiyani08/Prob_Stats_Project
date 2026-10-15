"""Smoke tests for the Streamlit app (app.py). Run: python3 -m pytest tests -q"""
import json
import os

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")


def _run(**setup):
    at = st_testing.AppTest.from_file(APP, default_timeout=120)
    at.run()
    for widget, value in setup.items():
        getattr(at, widget)[0].set_value(value)
    if setup:
        at.run()
    assert not at.exception, at.exception
    return at


def test_default_view_matches_full_study():
    """Default settings (Wi-Fi, 1 hour, seed 2026) must reproduce the saved study numbers."""
    at = _run()
    R = json.load(open(os.path.join(ROOT, "results", "results.json")))
    lost = R["scenarios"]["S2_WiFi"]["descriptive"]["losses"]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Packets lost"] == f"{lost:,}"


def test_coin_flip_mode_runs():
    at = _run(toggle=True)
    assert any("independent" in md.value for md in at.markdown)


@pytest.mark.parametrize("network", [0, 2])
def test_other_networks_run(network):
    at = st_testing.AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.radio[0].set_value(at.radio[0].options[network]).run()
    assert not at.exception, at.exception
