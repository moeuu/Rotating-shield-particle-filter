"""Launch private RA-L runtime and isolated controller processes."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import subprocess
import tempfile


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
    pf_config_path: Path,
    control_policy_path: Path,
    expected_control_policy_sha256: str,
    pf_output_dir: Path,
    pf_seed: int,
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
        "--config",
        pf_config_path.expanduser().resolve().as_posix(),
        "--control-policy",
        control_policy_path.expanduser().resolve().as_posix(),
        "--expected-control-policy-sha256",
        expected_control_policy_sha256,
        "--output-dir",
        pf_output_dir.expanduser().resolve().as_posix(),
        "--profile",
        "pf_strict",
        "--seed",
        str(pf_seed),
    ]


def run_isolated_ral_session(
    *,
    runtime_root: Path,
    scenario_path: Path,
    pf_config_path: Path,
    control_policy_path: Path,
    expected_control_policy_sha256: str,
    pf_output_dir: Path,
    pf_seed: int,
) -> int:
    """Run private acquisition and PF control with an opaque socket boundary."""
    runtime_root = runtime_root.expanduser().resolve()
    scenario_path = scenario_path.expanduser().resolve()
    if not runtime_root.is_dir() or not scenario_path.is_file():
        raise FileNotFoundError("Runtime root and private scenario must exist.")
    private_runtime_root = runtime_root / "private_runs"
    if not scenario_path.is_relative_to(private_runtime_root):
        raise ValueError("RA-L scenario must remain below runtime/private_runs.")
    with tempfile.TemporaryDirectory(prefix="ral-adaptive-session-") as directory:
        socket_path = Path(directory) / "runtime.sock"
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
        ]
        runtime_process = subprocess.Popen(runtime_command, cwd=runtime_root)
        try:
            controller_command = _controller_command(
                socket_path=socket_path,
                runtime_root=runtime_root,
                pf_config_path=pf_config_path,
                control_policy_path=control_policy_path,
                expected_control_policy_sha256=expected_control_policy_sha256,
                pf_output_dir=pf_output_dir,
                pf_seed=pf_seed,
            )
            completed = subprocess.run(controller_command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                return int(completed.returncode)
            return_code = runtime_process.wait(timeout=30.0)
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, runtime_command)
            return 0
        finally:
            _terminate_runtime(runtime_process)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse one private RA-L orchestration entry and run isolated processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--pf-config", type=Path, required=True)
    parser.add_argument("--control-policy", type=Path, required=True)
    parser.add_argument("--expected-control-policy-sha256", required=True)
    parser.add_argument("--pf-output-dir", type=Path, required=True)
    parser.add_argument("--pf-seed", type=int, required=True)
    args = parser.parse_args(None if argv is None else list(argv))
    return run_isolated_ral_session(
        runtime_root=args.runtime_root,
        scenario_path=args.scenario,
        pf_config_path=args.pf_config,
        control_policy_path=args.control_policy,
        expected_control_policy_sha256=args.expected_control_policy_sha256,
        pf_output_dir=args.pf_output_dir,
        pf_seed=args.pf_seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
