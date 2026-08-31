"""Launch private RA-L runtime and isolated controller processes."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import subprocess
import tempfile

from evaluation.completed_run import evaluate_completed_pf_run
from runtime.artifacts import atomic_write_json


ROOT = Path(__file__).resolve().parents[3]


def _terminate_runtime(process: subprocess.Popen[bytes]) -> None:
    """Stop an unfinished private runtime process within a bounded interval."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _controller_command(
    *,
    socket_path: Path,
    runtime_root: Path,
    cui_truth_overlay_socket_path: Path,
    pf_config_path: Path,
    control_policy_path: Path,
    expected_control_policy_sha256: str,
    pf_output_dir: Path,
    pf_seed: int,
    station_stop_request_path: Path,
) -> list[str]:
    """Build the PF-process command from truth-free inputs only."""
    return [
        "uv",
        "run",
        "--directory",
        ROOT.as_posix(),
        "python",
        "-m",
        "baselines.ral_ablation.live_controller",
        "--session-socket",
        socket_path.as_posix(),
        "--runtime-root",
        runtime_root.as_posix(),
        "--cui-truth-overlay-socket",
        cui_truth_overlay_socket_path.as_posix(),
        "--config",
        pf_config_path.expanduser().resolve().as_posix(),
        "--control-policy",
        control_policy_path.expanduser().resolve().as_posix(),
        "--expected-control-policy-sha256",
        expected_control_policy_sha256,
        "--output-dir",
        pf_output_dir.expanduser().resolve().as_posix(),
        "--station-stop-request",
        station_stop_request_path.expanduser().resolve().as_posix(),
        "--profile",
        "pf_strict",
        "--seed",
        str(pf_seed),
    ]


def run_isolated_ral_session(
    *,
    runtime_root: Path,
    scenario_path: Path,
    truth_manifest_path: Path,
    pf_config_path: Path,
    control_policy_path: Path,
    expected_control_policy_sha256: str,
    pf_output_dir: Path,
    pf_seed: int,
) -> int:
    """Run private acquisition, isolated PF control, and post-run evaluation."""
    runtime_root = runtime_root.expanduser().resolve()
    scenario_path = scenario_path.expanduser().resolve()
    truth_manifest_path = truth_manifest_path.expanduser().resolve()
    if (
        not runtime_root.is_dir()
        or not scenario_path.is_file()
        or not truth_manifest_path.is_file()
    ):
        raise FileNotFoundError(
            "Runtime root, private scenario, and truth manifest must exist."
        )
    private_runtime_root = runtime_root / "private_runs"
    if not scenario_path.is_relative_to(
        private_runtime_root
    ) or not truth_manifest_path.is_relative_to(private_runtime_root):
        raise ValueError(
            "RA-L scenario and truth manifest must remain below runtime/private_runs."
        )
    evaluation_output = (
        truth_manifest_path.parent.parent
        / "evaluations"
        / f"{truth_manifest_path.stem}.json"
    )
    if evaluation_output.exists():
        raise FileExistsError(
            "Refusing to replace an existing post-run evaluation: "
            f"{evaluation_output}"
        )
    stop_request_path = (
        truth_manifest_path.parent.parent
        / "stop_requests"
        / f"{truth_manifest_path.stem}.stop"
    )
    stop_request_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    stop_request_path.parent.chmod(0o700)
    if stop_request_path.exists() or stop_request_path.is_symlink():
        raise FileExistsError(
            "Refusing a stale RA-L station-stop request: "
            f"{stop_request_path}"
        )
    print(
        "RA-L graceful station-stop sentinel (eligible after 10 completed "
        f"stations): {stop_request_path}"
    )
    with tempfile.TemporaryDirectory(prefix="ral-adaptive-session-") as directory:
        socket_path = Path(directory) / "runtime.sock"
        cui_truth_overlay_socket_path = Path(directory) / "cui-truth.sock"
        runtime_command = [
            "uv",
            "run",
            "--directory",
            runtime_root.as_posix(),
            "rotating-shield-sim",
            "serve-adaptive-session-socket",
            scenario_path.as_posix(),
            "--socket-path",
            socket_path.as_posix(),
            "--cui-truth-overlay-socket-path",
            cui_truth_overlay_socket_path.as_posix(),
        ]
        runtime_process = subprocess.Popen(runtime_command, cwd=runtime_root)
        try:
            controller_command = _controller_command(
                socket_path=socket_path,
                runtime_root=runtime_root,
                cui_truth_overlay_socket_path=cui_truth_overlay_socket_path,
                pf_config_path=pf_config_path,
                control_policy_path=control_policy_path,
                expected_control_policy_sha256=expected_control_policy_sha256,
                pf_output_dir=pf_output_dir,
                pf_seed=pf_seed,
                station_stop_request_path=stop_request_path,
            )
            completed = subprocess.run(controller_command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                return int(completed.returncode)
            return_code = runtime_process.wait(timeout=30.0)
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, runtime_command)
            evaluation = evaluate_completed_pf_run(
                result_path=pf_output_dir / "closed_loop_result.json",
                posterior_path=pf_output_dir / "pf_posterior.json",
                evaluation_input_path=(
                    pf_output_dir / "pf_post_run_evaluation_input.json"
                ),
                truth_manifest_path=truth_manifest_path,
            )
            atomic_write_json(evaluation_output, evaluation)
            return 0
        finally:
            _terminate_runtime(runtime_process)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse one private RA-L orchestration entry and run isolated processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--truth-manifest", type=Path, required=True)
    parser.add_argument("--pf-config", type=Path, required=True)
    parser.add_argument("--control-policy", type=Path, required=True)
    parser.add_argument("--expected-control-policy-sha256", required=True)
    parser.add_argument("--pf-output-dir", type=Path, required=True)
    parser.add_argument("--pf-seed", type=int, required=True)
    args = parser.parse_args(None if argv is None else list(argv))
    return run_isolated_ral_session(
        runtime_root=args.runtime_root,
        scenario_path=args.scenario,
        truth_manifest_path=args.truth_manifest,
        pf_config_path=args.pf_config,
        control_policy_path=args.control_policy,
        expected_control_policy_sha256=args.expected_control_policy_sha256,
        pf_output_dir=args.pf_output_dir,
        pf_seed=args.pf_seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
