"""Unit tests for the conformal-forecast driver's data-free helpers.

`run_deepar_conformal.py` also has loaders that hit the network (CO2, Nile,
Sunspots) and a model that needs GluonTS; only the parts that must be
deterministic and local are covered here.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent / "run_deepar_conformal.py"

spec = importlib.util.spec_from_file_location("run_deepar_conformal", MODULE)
conf = importlib.util.module_from_spec(spec)
sys.modules.setdefault("run_deepar_conformal", conf)
spec.loader.exec_module(conf)


def test_coverage_counts_points_inside_the_interval():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lo = np.array([0.5, 1.5, 2.5, 3.5])
    hi = np.array([1.5, 2.5, 3.5, 4.5])
    assert conf.coverage(y, lo, hi) == 1.0
    assert conf.coverage(y, lo * 0, hi * 0) == 0.0


def test_coverage_is_a_fraction_not_a_count():
    y = np.array([0.0, 10.0])
    assert conf.coverage(y, np.array([-1.0, -1.0]), np.array([0.0, 0.0])) == 0.5


def test_synthetic_series_is_reproducible_and_weekly():
    a = conf.load_synthetic()
    b = conf.load_synthetic()
    assert len(a) == 200
    pd_started = a["date"].iloc[1] - a["date"].iloc[0]
    assert pd_started.days == 7
    np.testing.assert_allclose(a["value"].to_numpy(), b["value"].to_numpy())


def test_dataset_registry_declares_the_four_documented_series():
    assert set(conf.DATASETS) == {"CO2", "Nile", "Sunspots", "Synthetic"}
    # (loader, horizon, frequency) for every entry, and the loaders are callable
    for name, (loader, horizon, freq) in conf.DATASETS.items():
        assert callable(loader), name
        assert isinstance(horizon, int) and horizon > 0, name
        assert isinstance(freq, str) and freq, name
    # the only offline dataset in the registry
    assert conf.DATASETS["Synthetic"][0] is conf.load_synthetic


def test_nile_yearly_start_maps_to_the_gluonts_alias():
    assert conf.FREQ_MAP == {"YS": "Y"}


def test_seed_is_recorded_for_the_run():
    assert conf.SEED == 42
