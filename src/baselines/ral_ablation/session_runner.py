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
    pf_output_dir: Path,
    pf_seed: int,
    resume_stage_path: Path | None = None,
    resume_compatibility_path: Path | None = None,
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
        if resume_stage_path is not None:
            runtime_command.extend(
                ("--resume-stage", resume_stage_path.expanduser().resolve().as_posix())
            )
        if resume_compatibility_path is not None:
            if resume_stage_path is None:
                raise ValueError("resume compatibility requires a resume stage.")
            runtime_command.extend(
                (
                    "--resume-compatibility",
                    resume_compatibility_path.expanduser().resolve().as_posix(),
                )
            )
        runtime_process = subprocess.Popen(runtime_command, cwd=runtime_root)
        try:
            controller_command = _controller_command(
                socket_path=socket_path,
                runtime_root=runtime_root,
                pf_config_path=pf_config_path,
                control_policy_path=control_policy_path,
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
    parser.add_argument("--pf-output-dir", type=Path, required=True)
    parser.add_argument("--pf-seed", type=int, required=True)
    parser.add_argument("--resume-stage", type=Path, default=None)
    parser.add_argument("--resume-compatibility", type=Path, default=None)
    args = parser.parse_args(None if argv is None else list(argv))
    return run_isolated_ral_session(
        runtime_root=args.runtime_root,
        scenario_path=args.scenario,
        pf_config_path=args.pf_config,
        control_policy_path=args.control_policy,
        pf_output_dir=args.pf_output_dir,
        pf_seed=args.pf_seed,
        resume_stage_path=args.resume_stage,
        resume_compatibility_path=args.resume_compatibility,
    )


if __name__ == "__main__":
    raise SystemExit(main())
