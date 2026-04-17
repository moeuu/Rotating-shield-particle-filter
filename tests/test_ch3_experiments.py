"""End-to-end smoke test for Chapter 3 experiment runner."""

import json
from dataclasses import replace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_ch3_results import _aggregate
from scripts.run_ch3_experiments import default_scenarios, run_experiments


def test_run_ch3_experiments_smoke(tmp_path: Path) -> None:
    """Run a reduced set of scenarios and confirm logs are produced."""
    scenario = replace(
        default_scenarios()[0],
        name="single_cs_shielded_test",
        observation_mu_by_isotope={"Cs-137": 0.625, "Co-60": 0.75},
        mismatch_label="observation_mu_plus25",
    )
    scenarios = [scenario]
    run_experiments(output_dir=tmp_path, seeds=[0], scenarios=scenarios)
    log_file = tmp_path / f"{scenario.name}.jsonl"
    assert log_file.exists()
    content = log_file.read_text().strip().splitlines()
    assert content
    entry = json.loads(content[0])
    assert "position_error" in entry
    assert "iso_accuracy" in entry
    assert entry["trial_success"] in {True, False}
    assert entry["position_target_m"] == 0.5
    assert entry["mismatch_label"] == "observation_mu_plus25"
    assert entry["scenario"] == scenario.name
    assert entry["observation_mu_by_isotope"]["Cs-137"] == pytest.approx(0.625)
    assert "isotope_metrics" in entry


def test_aggregate_reports_success_and_iqr() -> None:
    """Scenario aggregation should expose robust Monte Carlo summaries."""
    summary = _aggregate(
        [
            {
                "scenario": "demo",
                "position_error": 0.2,
                "strength_error": 1.0,
                "iso_accuracy": 1.0,
                "trial_success": True,
                "position_within_target": True,
                "position_target_m": 0.5,
                "fp_count": 0,
                "fn_count": 0,
                "global_uncertainty": 0.1,
                "mismatch_label": "matched",
                "estimator_mu_by_isotope": {"Cs-137": 0.5},
                "observation_mu_by_isotope": {"Cs-137": 0.5},
            },
            {
                "scenario": "demo",
                "position_error": 0.6,
                "strength_error": 3.0,
                "iso_accuracy": 0.5,
                "trial_success": False,
                "position_within_target": False,
                "position_target_m": 0.5,
                "fp_count": 1,
                "fn_count": 1,
                "global_uncertainty": 0.4,
                "mismatch_label": "matched",
                "estimator_mu_by_isotope": {"Cs-137": 0.5},
                "observation_mu_by_isotope": {"Cs-137": 0.5},
            },
        ]
    )
    metrics = summary["demo"]
    assert metrics["num_trials"] == 2
    assert metrics["trial_success_rate"] == pytest.approx(0.5)
    assert metrics["position_q25"] == pytest.approx(0.3)
    assert metrics["position_q50"] == pytest.approx(0.4)
    assert metrics["position_q75"] == pytest.approx(0.5)
    assert metrics["iso_mean"] == pytest.approx(0.75)
    assert metrics["mismatch_label"] == "matched"
