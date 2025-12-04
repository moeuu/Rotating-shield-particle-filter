"""End-to-end smoke test for Chapter 3 experiment runner."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ch3_experiments import default_scenarios, run_experiments


def test_run_ch3_experiments_smoke(tmp_path: Path) -> None:
    """Run a reduced set of scenarios and confirm logs are produced."""
    scenarios = default_scenarios()[:1]
    run_experiments(output_dir=tmp_path, seeds=[0], scenarios=scenarios)
    log_file = tmp_path / f"{scenarios[0].name}.jsonl"
    assert log_file.exists()
    content = log_file.read_text().strip().splitlines()
    assert content
    # basic fields presence
    import json

    entry = json.loads(content[0])
    assert "position_error" in entry
    assert "iso_accuracy" in entry
    assert entry["scenario"] == scenarios[0].name
